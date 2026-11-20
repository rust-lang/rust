#!/usr/bin/env python3

"""
The Rust toolchain test runner for Fuchsia.

For instructions on running the compiler test suite, see
https://doc.rust-lang.org/stable/rustc/platform-support/fuchsia.html#aarch64-unknown-fuchsia-and-x86_64-unknown-fuchsia
"""

import argparse
import glob
import io
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional


def check_call_with_logging(
    args, *, stdout_handler, stderr_handler, check=True, text=True, **kwargs
):
    stdout_handler(f"Subprocess: {shlex.join(str(arg) for arg in args)}")

    with subprocess.Popen(
        args,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    ) as process:
        with ThreadPoolExecutor(max_workers=2) as executor:

            def exhaust_pipe(handler, pipe):
                for line in pipe:
                    handler(line.rstrip())

            executor_out = executor.submit(exhaust_pipe, stdout_handler, process.stdout)
            executor_err = executor.submit(exhaust_pipe, stderr_handler, process.stderr)
            executor_out.result()
            executor_err.result()
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(retcode, process.args)
    return subprocess.CompletedProcess(process.args, retcode)


def check_output_with_logging(
    args, *, stdout_handler, stderr_handler, check=True, text=True, **kwargs
):
    stdout_handler(f"Subprocess: {shlex.join(str(arg) for arg in args)}")

    buf = io.StringIO()

    with subprocess.Popen(
        args,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    ) as process:
        with ThreadPoolExecutor(max_workers=2) as executor:

            def exhaust_stdout(handler, buf, pipe):
                for line in pipe:
                    handler(line.rstrip())
                    buf.write(line)
                    buf.write("\n")

            def exhaust_stderr(handler, pipe):
                for line in pipe:
                    handler(line.rstrip())

            executor_out = executor.submit(
                exhaust_stdout, stdout_handler, buf, process.stdout
            )
            executor_err = executor.submit(
                exhaust_stderr, stderr_handler, process.stderr
            )
            executor_out.result()
            executor_err.result()
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(retcode, process.args)

    return buf.getvalue()


def atomic_link(link: Path, target: Path):
    link_dir = link.parent
    os.makedirs(link_dir, exist_ok=True)
    link_file = link.name
    tmp_file = link_dir.joinpath(link_file + "_tmp")
    os.link(target, tmp_file)
    try:
        os.rename(tmp_file, link)
    except Exception as e:
        raise e
    finally:
        if tmp_file.exists():
            os.remove(tmp_file)


@dataclass
class TestEnvironment:
    rust_build_dir: Path
    sdk_dir: Path
    target: str
    toolchain_dir: Path
    local_pb_path: Optional[Path]
    use_local_pb: bool
    verbose: bool = False

    env_logger = logging.getLogger("env")
    subprocess_logger = logging.getLogger("env.subprocess")
    __tmp_dir = None

    @staticmethod
    def tmp_dir() -> Path:
        if TestEnvironment.__tmp_dir:
            return TestEnvironment.__tmp_dir
        tmp_dir = os.environ.get("TEST_TOOLCHAIN_TMP_DIR")
        if tmp_dir is not None:
            TestEnvironment.__tmp_dir = Path(tmp_dir).absolute()
        else:
            TestEnvironment.__tmp_dir = Path(__file__).parent.joinpath("tmp~")
        return TestEnvironment.__tmp_dir

    @staticmethod
    def triple_to_arch(triple) -> str:
        if "x86_64" in triple:
            return "x64"
        elif "aarch64" in triple:
            return "arm64"
        else:
            raise Exception(f"Unrecognized target triple {triple}")

    @classmethod
    def env_file_path(cls) -> Path:
        return cls.tmp_dir().joinpath("test_env.json")

    @classmethod
    def from_args(cls, args):
        local_pb_path = args.local_product_bundle_path
        if local_pb_path is not None:
            local_pb_path = Path(local_pb_path).absolute()

        return cls(
            rust_build_dir=Path(args.rust_build).absolute(),
            sdk_dir=Path(args.sdk).absolute(),
            target=args.target,
            toolchain_dir=Path(args.toolchain_dir).absolute(),
            local_pb_path=local_pb_path,
            use_local_pb=args.use_local_product_bundle_if_exists,
            verbose=args.verbose,
        )

    @classmethod
    def read_from_file(cls):
        with open(cls.env_file_path(), encoding="utf-8") as f:
            test_env = json.load(f)
            local_pb_path = test_env["local_pb_path"]
            if local_pb_path is not None:
                local_pb_path = Path(local_pb_path)

            return cls(
                rust_build_dir=Path(test_env["rust_build_dir"]),
                sdk_dir=Path(test_env["sdk_dir"]),
                target=test_env["target"],
                toolchain_dir=Path(test_env["toolchain_dir"]),
                local_pb_path=local_pb_path,
                use_local_pb=test_env["use_local_pb"],
                verbose=test_env["verbose"],
            )

    def build_id(self, binary):
        llvm_readelf = Path(self.toolchain_dir).joinpath("bin", "llvm-readelf")
        process = subprocess.run(
            args=[
                llvm_readelf,
                "-n",
                "--elf-output-style=JSON",
                binary,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.returncode:
            e = f"llvm-readelf failed for binary {binary} with output {process.stdout}"
            self.env_logger.error(e)
            raise Exception(e)

        try:
            elf_output = json.loads(process.stdout)
        except Exception as e:
            e.add_note(f"Failed to read JSON from llvm-readelf for binary {binary}")
            e.add_note(f"stdout: {process.stdout}")
            raise

        try:
            note_sections = elf_output[0]["NoteSections"]
        except Exception as e:
            e.add_note(
                f'Failed to read "NoteSections" from llvm-readelf for binary {binary}'
            )
            e.add_note(f"elf_output: {elf_output}")
            raise

        for entry in note_sections:
            try:
                note_section = entry["NoteSection"]
                if note_section["Name"] == ".note.gnu.build-id":
                    return note_section["Notes"][0]["Build ID"]
            except Exception as e:
                e.add_note(
                    f'Failed to read ".note.gnu.build-id" from NoteSections \
                        entry in llvm-readelf for binary {binary}'
                )
                e.add_note(f"NoteSections: {note_sections}")
                raise
        raise Exception(f"Build ID not found for binary {binary}")

    def generate_buildid_dir(
        self,
        binary: Path,
        build_id_dir: Path,
        build_id: str,
        log_handler: logging.Logger,
    ):
        os.makedirs(build_id_dir, exist_ok=True)
        suffix = ".debug"
        # Hardlink the original binary
        build_id_prefix_dir = build_id_dir.joinpath(build_id[:2])
        unstripped_binary = build_id_prefix_dir.joinpath(build_id[2:] + suffix)
        build_id_prefix_dir.mkdir(parents=True, exist_ok=True)
        atomic_link(unstripped_binary, binary)
        assert unstripped_binary.exists()
        stripped_binary = unstripped_binary.with_suffix("")
        llvm_objcopy = Path(self.toolchain_dir).joinpath("bin", "llvm-objcopy")
        strip_mode = "--strip-sections"
        check_call_with_logging(
            [
                llvm_objcopy,
                strip_mode,
                unstripped_binary,
                stripped_binary,
            ],
            stdout_handler=log_handler.info,
            stderr_handler=log_handler.error,
        )
        return stripped_binary

    def write_to_file(self):
        with open(self.env_file_path(), "w", encoding="utf-8") as f:
            local_pb_path = self.local_pb_path
            if local_pb_path is not None:
                local_pb_path = str(local_pb_path)

            json.dump(
                {
                    "rust_build_dir": str(self.rust_build_dir),
                    "sdk_dir": str(self.sdk_dir),
                    "target": self.target,
                    "toolchain_dir": str(self.toolchain_dir),
                    "local_pb_path": local_pb_path,
                    "use_local_pb": self.use_local_pb,
                    "verbose": self.verbose,
                },
                f,
            )

    def setup_logging(self, log_to_file=False):
        fs = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
        if log_to_file:
            logfile_handler = logging.FileHandler(self.tmp_dir().joinpath("log"))
            logfile_handler.setLevel(logging.DEBUG)
            logfile_handler.setFormatter(fs)
            logging.getLogger().addHandler(logfile_handler)
        logging.getLogger().setLevel(logging.DEBUG)

    @property
    def package_server_log_path(self) -> Path:
        return self.tmp_dir().joinpath(f"repo_{self.TEST_REPO_NAME}.log")

    @property
    def emulator_log_path(self) -> Path:
        return self.tmp_dir().joinpath("emulator_log")

    @property
    def packages_dir(self) -> Path:
        return self.tmp_dir().joinpath("packages")

    @property
    def output_dir(self) -> Path:
        return self.tmp_dir().joinpath("output")

    def read_sdk_version(self):
        meta_json_path = Path(self.sdk_dir).joinpath("meta", "manifest.json")
        with open(meta_json_path, encoding="utf-8") as f:
            meta_json = json.load(f)
            return meta_json["id"]

    TEST_REPO_NAME: ClassVar[str] = "rust-testing"

    def repo_dir(self) -> Path:
        return self.tmp_dir().joinpath(self.TEST_REPO_NAME)

    def libs_dir(self) -> Path:
        return self.rust_build_dir.joinpath(
            "host",
            "stage2",
            "lib",
        )

    def rustlibs_dir(self) -> Path:
        return self.libs_dir().joinpath(
            "rustlib",
            self.target,
            "lib",
        )

    def sdk_arch(self):
        machine = platform.machine()
        if machine == "x86_64":
            return "x64"
        if machine == "arm":
            return "a64"
        raise Exception(f"Unrecognized host architecture {machine}")

    def tool_path(self, tool) -> Path:
        return Path(self.sdk_dir).joinpath("tools", self.sdk_arch(), tool)

    def host_arch_triple(self):
        machine = platform.machine()
        if machine == "x86_64":
            return "x86_64-unknown-linux-gnu"
        if machine == "arm":
            return "aarch64-unknown-linux-gnu"
        raise Exception(f"Unrecognized host architecture {machine}")

    def zxdb_script_path(self) -> Path:
        return Path(self.tmp_dir(), "zxdb_script")

    @property
    def ffx_daemon_log_path(self):
        return self.tmp_dir().joinpath("ffx_daemon_log")

    @property
    def ffx_isolate_dir(self):
        return self.tmp_dir().joinpath("ffx_isolate")

    @property
    def home_dir(self):
        return self.tmp_dir().joinpath("user-home")

    def start_ffx_isolation(self):
        # Most of this is translated directly from ffx's isolate library
        os.mkdir(self.ffx_isolate_dir)
        os.mkdir(self.home_dir)

        ffx_path = self.tool_path("ffx")
        ffx_env = self.ffx_cmd_env()

        # Start ffx daemon
        # We want this to be a long-running process that persists after the script finishes
        # pylint: disable=consider-using-with
        with open(
            self.ffx_daemon_log_path, "w", encoding="utf-8"
        ) as ffx_daemon_log_file:
            subprocess.Popen(
                [
                    ffx_path,
                    "daemon",
                    "start",
                ],
                env=ffx_env,
                stdout=ffx_daemon_log_file,
                stderr=ffx_daemon_log_file,
            )

        # Disable analytics
        check_call_with_logging(
            [
                ffx_path,
                "config",
                "analytics",
                "disable",
            ],
            env=ffx_env,
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        # Set configs
        configs = {
            "log.enabled": "true",
            "log.dir": self.tmp_dir(),
            "test.is_isolated": "true",
            "test.experimental_structured_output": "true",
        }
        for key, value in configs.items():
            check_call_with_logging(
                [
                    ffx_path,
                    "config",
                    "set",
                    key,
                    value,
                ],
                env=ffx_env,
                stdout_handler=self.subprocess_logger.debug,
                stderr_handler=self.subprocess_logger.debug,
            )

    def ffx_cmd_env(self):
        return {
            "HOME": self.home_dir,
            "FFX_ISOLATE_DIR": self.ffx_isolate_dir,
            # We want to use our own specified temp directory
            "TMP": self.tmp_dir(),
            "TEMP": self.tmp_dir(),
            "TMPDIR": self.tmp_dir(),
            "TEMPDIR": self.tmp_dir(),
        }

    def stop_ffx_isolation(self):
        check_call_with_logging(
            [
                self.tool_path("ffx"),
                "daemon",
                "stop",
            ],
            env=self.ffx_cmd_env(),
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

    def start(self):
        """Sets up the testing environment and prepares to run tests.

        Args:
            args: The command-line arguments to this command.

        During setup, this function will:
        - Locate necessary shared libraries
        - Create a new temp directory (this is where all temporary files are stored)
        - Start an emulator
        - Start an update server
        - Create a new package repo and register it with the emulator
        - Write test environment settings to a temporary file
        """

        # Initialize temp directory
        os.makedirs(self.tmp_dir(), exist_ok=True)
        if len(os.listdir(self.tmp_dir())) != 0:
            raise Exception(f"Temp directory is not clean (in {self.tmp_dir()})")
        self.setup_logging(log_to_file=True)
        os.mkdir(self.output_dir)

        ffx_path = self.tool_path("ffx")
        ffx_env = self.ffx_cmd_env()

        # Start ffx isolation
        self.env_logger.info("Starting ffx isolation...")
        self.start_ffx_isolation()

        # Stop any running emulators (there shouldn't be any)
        check_call_with_logging(
            [
                ffx_path,
                "emu",
                "stop",
                "--all",
            ],
            env=ffx_env,
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        if not self.local_pb_path:
            self.local_pb_path = os.path.join(self.tmp_dir(), "local_pb")
        else:
            self.local_pb_path = os.path.abspath(self.local_pb_path)

        if self.use_local_pb and os.path.exists(self.local_pb_path):
            self.env_logger.info(
                'Using existing emulator image at "%s"' % self.local_pb_path
            )
        else:
            shutil.rmtree(self.local_pb_path, ignore_errors=True)

            # Look up the product bundle transfer manifest.
            self.env_logger.info("Looking up the product bundle transfer manifest...")
            product_name = "minimal." + self.triple_to_arch(self.target)
            sdk_version = self.read_sdk_version()

            output = check_output_with_logging(
                [
                    ffx_path,
                    "--machine",
                    "json",
                    "product",
                    "lookup",
                    product_name,
                    sdk_version,
                    "--base-url",
                    "gs://fuchsia/development/" + sdk_version,
                ],
                env=ffx_env,
                stdout_handler=self.subprocess_logger.debug,
                stderr_handler=self.subprocess_logger.debug,
            )

            try:
                transfer_manifest_url = json.loads(output)["transfer_manifest_url"]
            except Exception as e:
                print(e)
                raise Exception("Unable to parse transfer manifest") from e

            # Download the product bundle.
            self.env_logger.info("Downloading the product bundle...")
            check_call_with_logging(
                [
                    ffx_path,
                    "product",
                    "download",
                    transfer_manifest_url,
                    self.local_pb_path,
                ],
                env=ffx_env,
                stdout_handler=self.subprocess_logger.debug,
                stderr_handler=self.subprocess_logger.debug,
            )

        # Start emulator
        self.env_logger.info("Starting emulator...")

        # FIXME: condition --accel hyper on target arch matching host arch
        check_call_with_logging(
            [
                ffx_path,
                "emu",
                "start",
                self.local_pb_path,
                "--headless",
                "--log",
                self.emulator_log_path,
                "--net",
                "auto",
                "--accel",
                "auto",
            ],
            env=ffx_env,
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        # Create new package repo
        self.env_logger.info("Creating package repo...")
        check_call_with_logging(
            [
                ffx_path,
                "repository",
                "create",
                self.repo_dir(),
            ],
            env=ffx_env,
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        check_call_with_logging(
            [
                ffx_path,
                "repository",
                "server",
                "start",
                "--background",
                "--address",
                "[::]:0",
                "--repo-path",
                self.repo_dir(),
                "--repository",
                self.TEST_REPO_NAME,
            ],
            env=ffx_env,
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        # Register with newly-started emulator
        check_call_with_logging(
            [
                ffx_path,
                "target",
                "repository",
                "register",
                "--repository",
                self.TEST_REPO_NAME,
            ],
            env=ffx_env,
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        # Write to file
        self.write_to_file()

        self.env_logger.info("Success! Your environment is ready to run tests.")

    # FIXME: shardify this
    # `facet` statement required for TCP testing via
    # protocol `fuchsia.posix.socket.Provider`. See
    # https://fuchsia.dev/fuchsia-src/development/testing/components/test_runner_framework?hl=en#legacy_non-hermetic_tests
    CML_TEMPLATE: ClassVar[str] = """
    {{
        program: {{
            runner: "elf_test_runner",
            binary: "bin/{exe_name}",
            forward_stderr_to: "log",
            forward_stdout_to: "log",
            environ: [{env_vars}
            ]
        }},
        capabilities: [
            {{ protocol: "fuchsia.test.Suite" }},
        ],
        expose: [
            {{
                protocol: "fuchsia.test.Suite",
                from: "self",
            }},
        ],
        use: [
            {{ storage: "data", path: "/data" }},
            {{ storage: "tmp", path: "/tmp" }},
            {{ protocol: [ "fuchsia.process.Launcher" ] }},
            {{ protocol: [ "fuchsia.posix.socket.Provider" ] }}
        ],
        facets: {{
            "fuchsia.test": {{ type: "system" }},
        }},
    }}
    """

    MANIFEST_TEMPLATE = """
    meta/package={package_dir}/meta/package
    meta/{package_name}.cm={package_dir}/meta/{package_name}.cm
    bin/{exe_name}={bin_path}
    lib/{libstd_name}={libstd_path}
    lib/ld.so.1={sdk_dir}/arch/{target_arch}/sysroot/dist/lib/ld.so.1
    lib/libfdio.so={sdk_dir}/arch/{target_arch}/dist/libfdio.so
    """

    TEST_ENV_VARS: ClassVar[List[str]] = [
        "TEST_EXEC_ENV",
        "RUST_MIN_STACK",
        "RUST_BACKTRACE",
        "RUST_NEWRT",
        "RUST_LOG",
        "RUST_TEST_THREADS",
    ]

    def run(self, args):
        """Runs the requested test in the testing environment.

        Args:
        args: The command-line arguments to this command.
        Returns:
        The return code of the test (0 for success, else failure).

        To run a test, this function will:
        - Create, compile, archive, and publish a test package
        - Run the test package on the emulator
        - Forward the test's stdout and stderr as this script's stdout and stderr
        """

        bin_path = Path(args.bin_path).absolute()

        # Find libstd and libtest
        libstd_paths = glob.glob(os.path.join(self.rustlibs_dir(), "libstd-*.so"))
        libtest_paths = glob.glob(os.path.join(self.rustlibs_dir(), "libtest-*.so"))

        if not libstd_paths:
            raise Exception(f"Failed to locate libstd (in {self.rustlibs_dir()})")

        base_name = os.path.basename(os.path.dirname(args.bin_path))
        exe_name = base_name.lower().replace(".", "_")
        build_id = self.build_id(bin_path)
        package_name = f"{exe_name}_{build_id}"

        package_dir = self.packages_dir.joinpath(package_name)
        package_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = package_dir.joinpath("meta")
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_package_path = meta_dir.joinpath("package")
        cml_path = meta_dir.joinpath(f"{package_name}.cml")
        cm_path = meta_dir.joinpath(f"{package_name}.cm")
        manifest_path = package_dir.joinpath(f"{package_name}.manifest")

        shared_libs = args.shared_libs[: args.n]
        arguments = args.shared_libs[args.n :]

        test_output_dir = self.output_dir.joinpath(package_name)

        # Clean and create temporary output directory
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        test_output_dir.mkdir(parents=True)

        # Open log file
        runner_logger = logging.getLogger(f"env.package.{package_name}")
        runner_logger.setLevel(logging.DEBUG)
        logfile_handler = logging.FileHandler(test_output_dir.joinpath("log"))
        logfile_handler.setLevel(logging.DEBUG)
        logfile_handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        runner_logger.addHandler(logfile_handler)

        runner_logger.info(f"Bin path: {bin_path}")
        runner_logger.info("Setting up package...")

        # Link binary to build-id dir and strip it.
        build_id_dir = self.tmp_dir().joinpath(".build-id")
        stripped_binary = self.generate_buildid_dir(
            binary=bin_path,
            build_id_dir=build_id_dir,
            build_id=build_id,
            log_handler=runner_logger,
        )
        runner_logger.info(f"Stripped Bin path: {stripped_binary}")

        runner_logger.info("Writing CML...")

        # Write and compile CML
        with open(cml_path, "w", encoding="utf-8") as cml:
            # Collect environment variables
            env_vars = ""
            for var_name in self.TEST_ENV_VARS:
                var_value = os.getenv(var_name)
                if var_value is not None:
                    env_vars += f'\n            "{var_name}={var_value}",'

            # Default to no backtrace for test suite
            if os.getenv("RUST_BACKTRACE") is None:
                env_vars += '\n            "RUST_BACKTRACE=0",'

            # Use /tmp as the test temporary directory
            env_vars += '\n            "RUST_TEST_TMPDIR=/tmp",'

            cml.write(self.CML_TEMPLATE.format(env_vars=env_vars, exe_name=exe_name))

        runner_logger.info("Compiling CML...")

        check_call_with_logging(
            [
                self.tool_path("cmc"),
                "compile",
                cml_path,
                "--includepath",
                ".",
                "--output",
                cm_path,
            ],
            stdout_handler=runner_logger.info,
            stderr_handler=runner_logger.warning,
        )

        runner_logger.info("Writing meta/package...")
        with open(meta_package_path, "w", encoding="utf-8") as f:
            json.dump({"name": package_name, "version": "0"}, f)

        runner_logger.info("Writing manifest...")

        # Write package manifest
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            manifest.write(
                self.MANIFEST_TEMPLATE.format(
                    bin_path=stripped_binary,
                    exe_name=exe_name,
                    package_dir=package_dir,
                    package_name=package_name,
                    target=self.target,
                    sdk_dir=self.sdk_dir,
                    libstd_name=os.path.basename(libstd_paths[0]),
                    libstd_path=libstd_paths[0],
                    target_arch=self.triple_to_arch(self.target),
                )
            )
            # `libtest`` was historically a shared library, but now seems to be (sometimes?)
            # statically linked. If we find it as a shared library, include it in the manifest.
            if libtest_paths:
                manifest.write(
                    f"lib/{os.path.basename(libtest_paths[0])}={libtest_paths[0]}\n"
                )
            for shared_lib in shared_libs:
                manifest.write(f"lib/{os.path.basename(shared_lib)}={shared_lib}\n")

        runner_logger.info("Determining API level...")
        out = check_output_with_logging(
            [
                self.tool_path("ffx"),
                "--machine",
                "json",
                "version",
            ],
            env=self.ffx_cmd_env(),
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )
        api_level = json.loads(out)["tool_version"]["api_level"]

        runner_logger.info("Compiling manifest...")

        check_call_with_logging(
            [
                self.tool_path("ffx"),
                "package",
                "build",
                manifest_path,
                "-o",
                package_dir,
                "--api-level",
                str(api_level),
            ],
            env=self.ffx_cmd_env(),
            stdout_handler=runner_logger.info,
            stderr_handler=runner_logger.warning,
        )

        runner_logger.info("Publishing package to repo...")

        # Publish package to repo
        check_call_with_logging(
            [
                self.tool_path("ffx"),
                "repository",
                "publish",
                "--package",
                os.path.join(package_dir, "package_manifest.json"),
                self.repo_dir(),
            ],
            env=self.ffx_cmd_env(),
            stdout_handler=runner_logger.info,
            stderr_handler=runner_logger.warning,
        )

        runner_logger.info("Running ffx test...")

        # Run test on emulator
        check_call_with_logging(
            [
                self.tool_path("ffx"),
                "test",
                "run",
                f"fuchsia-pkg://{self.TEST_REPO_NAME}/{package_name}#meta/{package_name}.cm",
                "--min-severity-logs",
                "TRACE",
                "--output-directory",
                test_output_dir,
                "--",
            ]
            + arguments,
            env=self.ffx_cmd_env(),
            check=False,
            stdout_handler=runner_logger.info,
            stderr_handler=runner_logger.warning,
        )

        runner_logger.info("Reporting test suite output...")

        # Read test suite output
        run_summary_path = test_output_dir.joinpath("run_summary.json")
        if not run_summary_path.exists():
            runner_logger.error("Failed to open test run summary")
            return 254

        with open(run_summary_path, encoding="utf-8") as f:
            run_summary = json.load(f)

        suite = run_summary["data"]["suites"][0]
        case = suite["cases"][0]

        return_code = 0 if case["outcome"] == "PASSED" else 1

        artifacts = case["artifacts"]
        artifact_dir = case["artifact_dir"]
        stdout_path = None
        stderr_path = None

        for path, artifact in artifacts.items():
            artifact_path = os.path.join(test_output_dir, artifact_dir, path)
            artifact_type = artifact["artifact_type"]

            if artifact_type == "STDERR":
                stderr_path = artifact_path
            elif artifact_type == "STDOUT":
                stdout_path = artifact_path

        if stdout_path is not None:
            if not os.path.exists(stdout_path):
                runner_logger.error(f"stdout file {stdout_path} does not exist.")
            else:
                with open(stdout_path, encoding="utf-8", errors="ignore") as f:
                    sys.stdout.write(f.read())
        if stderr_path is not None:
            if not os.path.exists(stderr_path):
                runner_logger.error(f"stderr file {stderr_path} does not exist.")
            else:
                with open(stderr_path, encoding="utf-8", errors="ignore") as f:
                    sys.stderr.write(f.read())

        runner_logger.info("Done!")
        return return_code

    def stop(self):
        """Shuts down and cleans up the testing environment.

        Args:
        args: The command-line arguments to this command.
        Returns:
        The return code of the test (0 for success, else failure).

        During cleanup, this function will stop the emulator, package server, and
        update server, then delete all temporary files. If an error is encountered
        while stopping any running processes, the temporary files will not be deleted.
        Passing --cleanup will force the process to delete the files anyway.
        """

        self.env_logger.debug("Reporting logs...")

        # Print test log files
        for test_dir in os.listdir(self.output_dir):
            log_path = os.path.join(self.output_dir, test_dir, "log")
            self.env_logger.debug(f"\n---- Logs for test '{test_dir}' ----\n")
            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8", errors="ignore") as log:
                    self.env_logger.debug(log.read())
            else:
                self.env_logger.debug("No logs found")

        # Print the emulator log
        self.env_logger.debug("\n---- Emulator logs ----\n")
        if os.path.exists(self.emulator_log_path):
            with open(self.emulator_log_path, encoding="utf-8") as log:
                self.env_logger.debug(log.read())
        else:
            self.env_logger.debug("No emulator logs found")

        # Print the package server log
        self.env_logger.debug("\n---- Package server log ----\n")
        if os.path.exists(self.package_server_log_path):
            with open(self.package_server_log_path, encoding="utf-8") as log:
                self.env_logger.debug(log.read())
        else:
            self.env_logger.debug("No package server log found")

        # Print the ffx daemon log
        self.env_logger.debug("\n---- ffx daemon log ----\n")
        if os.path.exists(self.ffx_daemon_log_path):
            with open(self.ffx_daemon_log_path, encoding="utf-8") as log:
                self.env_logger.debug(log.read())
        else:
            self.env_logger.debug("No ffx daemon log found")

        # Shut down the emulator
        self.env_logger.info("Stopping emulator...")
        check_call_with_logging(
            [
                self.tool_path("ffx"),
                "emu",
                "stop",
            ],
            env=self.ffx_cmd_env(),
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        # Stop the package server
        self.env_logger.info("Stopping package server...")
        check_call_with_logging(
            [
                self.tool_path("ffx"),
                "repository",
                "server",
                "stop",
                self.TEST_REPO_NAME,
            ],
            env=self.ffx_cmd_env(),
            stdout_handler=self.subprocess_logger.debug,
            stderr_handler=self.subprocess_logger.debug,
        )

        # Stop ffx isolation
        self.env_logger.info("Stopping ffx isolation...")
        self.stop_ffx_isolation()

    def cleanup(self):
        # Remove temporary files
        self.env_logger.info("Deleting temporary files...")
        shutil.rmtree(self.tmp_dir(), ignore_errors=True)

    def debug(self, args):
        command = [
            self.tool_path("ffx"),
            "debug",
            "connect",
            "--",
            "--build-id-dir",
            os.path.join(self.sdk_dir, ".build-id"),
        ]

        libs_build_id_path = os.path.join(self.libs_dir(), ".build-id")
        if os.path.exists(libs_build_id_path):
            # Add .build-id symbols if installed libs have been stripped into a
            # .build-id directory
            command += [
                "--build-id-dir",
                libs_build_id_path,
            ]
        else:
            # If no .build-id directory is detected, then assume that the shared
            # libs contain their debug symbols
            command += [
                f"--symbol-path={self.rust_dir}/lib/rustlib/{self.target}/lib",
            ]

        # Add rust source if it's available
        rust_src_map = None
        if args.rust_src is not None:
            # This matches the remapped prefix used by compiletest. There's no
            # clear way that we can determine this, so it's hard coded.
            rust_src_map = f"/rustc/FAKE_PREFIX={args.rust_src}"

        # Add fuchsia source if it's available
        fuchsia_src_map = None
        if args.fuchsia_src is not None:
            fuchsia_src_map = f"./../..={args.fuchsia_src}"

        # Load debug symbols for the test binary and automatically attach
        if args.test is not None:
            if args.rust_src is None:
                raise Exception(
                    "A Rust source path is required with the `test` argument"
                )

            test_name = os.path.splitext(os.path.basename(args.test))[0]

            build_dir = os.path.join(
                args.rust_src,
                "fuchsia-build",
                self.host_arch_triple(),
            )
            test_dir = os.path.join(
                build_dir,
                "test",
                os.path.dirname(args.test),
                test_name,
            )

            # The fake-test-src-base directory maps to the suite directory
            # e.g. tests/ui/foo.rs has a path of rust/fake-test-src-base/foo.rs
            fake_test_src_base = os.path.join(
                args.rust_src,
                "fake-test-src-base",
            )
            real_test_src_base = os.path.join(
                args.rust_src,
                "tests",
                args.test.split(os.path.sep)[0],
            )
            test_src_map = f"{fake_test_src_base}={real_test_src_base}"

            with open(self.zxdb_script_path(), mode="w", encoding="utf-8") as f:
                print(f"set source-map += {test_src_map}", file=f)

                if rust_src_map is not None:
                    print(f"set source-map += {rust_src_map}", file=f)

                if fuchsia_src_map is not None:
                    print(f"set source-map += {fuchsia_src_map}", file=f)

                print(f"attach {test_name[:31]}", file=f)

            command += [
                "--symbol-path",
                test_dir,
                "-S",
                self.zxdb_script_path(),
            ]

        # Add any other zxdb arguments the user passed
        if args.zxdb_args is not None:
            command += args.zxdb_args

        # Connect to the running emulator with zxdb
        subprocess.run(command, env=self.ffx_cmd_env(), check=False)

    def syslog(self, args):
        subprocess.run(
            [
                self.tool_path("ffx"),
                "log",
                "--since",
                "now",
            ],
            env=self.ffx_cmd_env(),
            check=False,
        )


def start(args):
    test_env = TestEnvironment.from_args(args)
    test_env.start()
    return 0


def run(args):
    test_env = TestEnvironment.read_from_file()
    test_env.setup_logging(log_to_file=True)
    return test_env.run(args)


def stop(args):
    test_env = TestEnvironment.read_from_file()
    test_env.setup_logging(log_to_file=False)
    test_env.stop()
    if not args.no_cleanup:
        test_env.cleanup()
    return 0


def cleanup(args):
    del args
    test_env = TestEnvironment.read_from_file()
    test_env.setup_logging(log_to_file=False)
    test_env.cleanup()
    return 0


def debug(args):
    test_env = TestEnvironment.read_from_file()
    test_env.debug(args)
    return 0


def syslog(args):
    test_env = TestEnvironment.read_from_file()
    test_env.setup_logging(log_to_file=True)
    test_env.syslog(args)
    return 0


def main():
    parser = argparse.ArgumentParser()

    def print_help(args):
        del args
        parser.print_help()
        return 0

    parser.set_defaults(func=print_help)

    subparsers = parser.add_subparsers(help="valid sub-commands")

    start_parser = subparsers.add_parser(
        "start", help="initializes the testing environment"
    )
    start_parser.add_argument(
        "--rust-build",
        help="the current compiler build directory (`$RUST_SRC/build` by default)",
        required=True,
    )
    start_parser.add_argument(
        "--sdk",
        help="the directory of the fuchsia SDK",
        required=True,
    )
    start_parser.add_argument(
        "--verbose",
        help="prints more output from executed processes",
        action="store_true",
    )
    start_parser.add_argument(
        "--target",
        help="the target platform to test",
        required=True,
    )
    start_parser.add_argument(
        "--toolchain-dir",
        help="the toolchain directory",
        required=True,
    )
    start_parser.add_argument(
        "--local-product-bundle-path",
        help="the path where the product-bundle should be downloaded to",
    )
    start_parser.add_argument(
        "--use-local-product-bundle-if-exists",
        help="if the product bundle already exists in the local path, use "
        "it instead of downloading it again",
        action="store_true",
    )
    start_parser.set_defaults(func=start)

    run_parser = subparsers.add_parser(
        "run", help="run a test in the testing environment"
    )
    run_parser.add_argument(
        "n", help="the number of shared libs passed along with the executable", type=int
    )
    run_parser.add_argument("bin_path", help="path to the binary to run")
    run_parser.add_argument(
        "shared_libs",
        help="the shared libs passed along with the binary",
        nargs=argparse.REMAINDER,
    )
    run_parser.set_defaults(func=run)

    stop_parser = subparsers.add_parser(
        "stop", help="shuts down and cleans up the testing environment"
    )
    stop_parser.add_argument(
        "--no-cleanup",
        default=False,
        action="store_true",
        help="don't delete temporary files after stopping",
    )
    stop_parser.set_defaults(func=stop)

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="deletes temporary files after the testing environment has been manually cleaned up",
    )
    cleanup_parser.set_defaults(func=cleanup)

    syslog_parser = subparsers.add_parser("syslog", help="prints the device syslog")
    syslog_parser.set_defaults(func=syslog)

    debug_parser = subparsers.add_parser(
        "debug",
        help="connect to the active testing environment with zxdb",
    )
    debug_parser.add_argument(
        "--rust-src",
        default=None,
        help="the path to the Rust source being tested",
    )
    debug_parser.add_argument(
        "--fuchsia-src",
        default=None,
        help="the path to the Fuchsia source",
    )
    debug_parser.add_argument(
        "--test",
        default=None,
        help="the path to the test to debug (e.g. ui/box/new.rs)",
    )
    debug_parser.add_argument(
        "zxdb_args",
        default=None,
        nargs=argparse.REMAINDER,
        help="any additional arguments to pass to zxdb",
    )
    debug_parser.set_defaults(func=debug)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
