#!/usr/bin/env python3

"""
The Rust toolchain test runner for Fuchsia.

For instructions on running the compiler test suite, see
https://doc.rust-lang.org/stable/rustc/platform-support/fuchsia.html#aarch64-unknown-fuchsia-and-x86_64-unknown-fuchsia
"""

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from typing import ClassVar, List, Optional


@dataclass
class TestEnvironment:
    rust_dir: str
    sdk_dir: str
    target: str
    package_server_pid: Optional[int] = None
    emu_addr: Optional[str] = None
    libstd_name: Optional[str] = None
    libtest_name: Optional[str] = None
    verbose: bool = False

    @staticmethod
    def tmp_dir():
        tmp_dir = os.environ.get("TEST_TOOLCHAIN_TMP_DIR")
        if tmp_dir is not None:
            return os.path.abspath(tmp_dir)
        return os.path.join(os.path.dirname(__file__), "tmp~")

    @staticmethod
    def triple_to_arch(triple):
        if "x86_64" in triple:
            return "x64"
        elif "aarch64" in triple:
            return "arm64"
        else:
            raise Exception(f"Unrecognized target triple {triple}")

    @classmethod
    def env_file_path(cls):
        return os.path.join(cls.tmp_dir(), "test_env.json")

    @classmethod
    def from_args(cls, args):
        return cls(
            os.path.abspath(args.rust),
            os.path.abspath(args.sdk),
            args.target,
            verbose=args.verbose,
        )

    @classmethod
    def read_from_file(cls):
        with open(cls.env_file_path(), encoding="utf-8") as f:
            test_env = json.loads(f.read())
            return cls(
                test_env["rust_dir"],
                test_env["sdk_dir"],
                test_env["target"],
                libstd_name=test_env["libstd_name"],
                libtest_name=test_env["libtest_name"],
                emu_addr=test_env["emu_addr"],
                package_server_pid=test_env["package_server_pid"],
                verbose=test_env["verbose"],
            )

    def write_to_file(self):
        with open(self.env_file_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.__dict__))

    def ssh_dir(self):
        return os.path.join(self.tmp_dir(), "ssh")

    def ssh_keyfile_path(self):
        return os.path.join(self.ssh_dir(), "fuchsia_ed25519")

    def ssh_authfile_path(self):
        return os.path.join(self.ssh_dir(), "fuchsia_authorized_keys")

    def vdl_output_path(self):
        return os.path.join(self.tmp_dir(), "vdl_output")

    def package_server_log_path(self):
        return os.path.join(self.tmp_dir(), "package_server_log")

    def emulator_log_path(self):
        return os.path.join(self.tmp_dir(), "emulator_log")

    def packages_dir(self):
        return os.path.join(self.tmp_dir(), "packages")

    def output_dir(self):
        return os.path.join(self.tmp_dir(), "output")

    TEST_REPO_NAME: ClassVar[str] = "rust-testing"

    def repo_dir(self):
        return os.path.join(self.tmp_dir(), self.TEST_REPO_NAME)

    def libs_dir(self):
        return os.path.join(
            self.rust_dir,
            "lib",
        )

    def rustlibs_dir(self):
        return os.path.join(
            self.libs_dir(),
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

    def tool_path(self, tool):
        return os.path.join(self.sdk_dir, "tools", self.sdk_arch(), tool)

    def host_arch_triple(self):
        machine = platform.machine()
        if machine == "x86_64":
            return "x86_64-unknown-linux-gnu"
        if machine == "arm":
            return "aarch64-unknown-linux-gnu"
        raise Exception(f"Unrecognized host architecture {machine}")

    def zxdb_script_path(self):
        return os.path.join(self.tmp_dir(), "zxdb_script")

    def log_info(self, msg):
        print(msg)

    def log_debug(self, msg):
        if self.verbose:
            print(msg)

    def subprocess_output(self):
        if self.verbose:
            return sys.stdout
        return subprocess.DEVNULL

    def ffx_daemon_log_path(self):
        return os.path.join(self.tmp_dir(), "ffx_daemon_log")

    def ffx_isolate_dir(self):
        return os.path.join(self.tmp_dir(), "ffx_isolate")

    def ffx_home_dir(self):
        return os.path.join(self.ffx_isolate_dir(), "user-home")

    def ffx_tmp_dir(self):
        return os.path.join(self.ffx_isolate_dir(), "tmp")

    def ffx_log_dir(self):
        return os.path.join(self.ffx_isolate_dir(), "log")

    def ffx_user_config_dir(self):
        return os.path.join(self.ffx_xdg_config_home(), "Fuchsia", "ffx", "config")

    def ffx_user_config_path(self):
        return os.path.join(self.ffx_user_config_dir(), "config.json")

    def ffx_xdg_config_home(self):
        if platform.system() == "Darwin":
            return os.path.join(self.ffx_home_dir(), "Library", "Preferences")
        return os.path.join(self.ffx_home_dir(), ".local", "share")

    def ffx_ascendd_path(self):
        return os.path.join(self.ffx_tmp_dir(), "ascendd")

    def start_ffx_isolation(self):
        # Most of this is translated directly from ffx's isolate library
        os.mkdir(self.ffx_isolate_dir())
        os.mkdir(self.ffx_home_dir())
        os.mkdir(self.ffx_tmp_dir())
        os.mkdir(self.ffx_log_dir())

        fuchsia_dir = os.path.join(self.ffx_home_dir(), ".fuchsia")
        os.mkdir(fuchsia_dir)

        fuchsia_debug_dir = os.path.join(fuchsia_dir, "debug")
        os.mkdir(fuchsia_debug_dir)

        metrics_dir = os.path.join(fuchsia_dir, "metrics")
        os.mkdir(metrics_dir)

        analytics_path = os.path.join(metrics_dir, "analytics-status")
        with open(analytics_path, "w", encoding="utf-8") as analytics_file:
            print("0", file=analytics_file)

        ffx_path = os.path.join(metrics_dir, "ffx")
        with open(ffx_path, "w", encoding="utf-8") as ffx_file:
            print("1", file=ffx_file)

        os.makedirs(self.ffx_user_config_dir())

        with open(
            self.ffx_user_config_path(), "w", encoding="utf-8"
        ) as config_json_file:
            user_config_for_test = {
                "log": {
                    "enabled": True,
                    "dir": self.ffx_log_dir(),
                },
                "overnet": {
                    "socket": self.ffx_ascendd_path(),
                },
                "ssh": {
                    "pub": self.ssh_authfile_path(),
                    "priv": self.ssh_keyfile_path(),
                },
                "test": {
                    "is_isolated": True,
                    "experimental_structured_output": True,
                },
            }
            print(json.dumps(user_config_for_test), file=config_json_file)

        ffx_env_path = os.path.join(self.ffx_user_config_dir(), ".ffx_env")
        with open(ffx_env_path, "w", encoding="utf-8") as ffx_env_file:
            ffx_env_config_for_test = {
                "user": self.ffx_user_config_path(),
                "build": None,
                "global": None,
            }
            print(json.dumps(ffx_env_config_for_test), file=ffx_env_file)

        # Start ffx daemon
        # We want this to be a long-running process that persists after the script finishes
        # pylint: disable=consider-using-with
        with open(
            self.ffx_daemon_log_path(), "w", encoding="utf-8"
        ) as ffx_daemon_log_file:
            subprocess.Popen(
                [
                    self.tool_path("ffx"),
                    "--config",
                    self.ffx_user_config_path(),
                    "daemon",
                    "start",
                ],
                env=self.ffx_cmd_env(),
                stdout=ffx_daemon_log_file,
                stderr=ffx_daemon_log_file,
            )

    def ffx_cmd_env(self):
        result = {
            "HOME": self.ffx_home_dir(),
            "XDG_CONFIG_HOME": self.ffx_xdg_config_home(),
            "ASCENDD": self.ffx_ascendd_path(),
            "FUCHSIA_SSH_KEY": self.ssh_keyfile_path(),
            # We want to use our own specified temp directory
            "TMP": self.tmp_dir(),
            "TEMP": self.tmp_dir(),
            "TMPDIR": self.tmp_dir(),
            "TEMPDIR": self.tmp_dir(),
        }

        return result

    def stop_ffx_isolation(self):
        subprocess.check_call(
            [
                self.tool_path("ffx"),
                "--config",
                self.ffx_user_config_path(),
                "daemon",
                "stop",
            ],
            env=self.ffx_cmd_env(),
            stdout=self.subprocess_output(),
            stderr=self.subprocess_output(),
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
        if not os.path.exists(self.tmp_dir()):
            os.mkdir(self.tmp_dir())
        elif len(os.listdir(self.tmp_dir())) != 0:
            raise Exception(f"Temp directory is not clean (in {self.tmp_dir()})")

        os.mkdir(self.ssh_dir())
        os.mkdir(self.output_dir())

        # Find libstd and libtest
        libstd_paths = glob.glob(os.path.join(self.rustlibs_dir(), "libstd-*.so"))
        libtest_paths = glob.glob(os.path.join(self.rustlibs_dir(), "libtest-*.so"))

        if not libstd_paths:
            raise Exception(f"Failed to locate libstd (in {self.rustlibs_dir()})")

        if not libtest_paths:
            raise Exception(f"Failed to locate libtest (in {self.rustlibs_dir()})")

        self.libstd_name = os.path.basename(libstd_paths[0])
        self.libtest_name = os.path.basename(libtest_paths[0])

        # Generate SSH keys for the emulator to use
        self.log_info("Generating SSH keys...")
        subprocess.check_call(
            [
                "ssh-keygen",
                "-N",
                "",
                "-t",
                "ed25519",
                "-f",
                self.ssh_keyfile_path(),
                "-C",
                "Generated by fuchsia-test-runner.py",
            ],
            stdout=self.subprocess_output(),
            stderr=self.subprocess_output(),
        )
        authfile_contents = subprocess.check_output(
            [
                "ssh-keygen",
                "-y",
                "-f",
                self.ssh_keyfile_path(),
            ],
            stderr=self.subprocess_output(),
        )
        with open(self.ssh_authfile_path(), "wb") as authfile:
            authfile.write(authfile_contents)

        # Start ffx isolation
        self.log_info("Starting ffx isolation...")
        self.start_ffx_isolation()

        # Start emulator (this will generate the vdl output)
        self.log_info("Starting emulator...")
        subprocess.check_call(
            [
                self.tool_path("fvdl"),
                "--sdk",
                "start",
                "--tuntap",
                "--headless",
                "--nointeractive",
                "--ssh",
                self.ssh_dir(),
                "--vdl-output",
                self.vdl_output_path(),
                "--emulator-log",
                self.emulator_log_path(),
                "--image-name",
                "qemu-" + self.triple_to_arch(self.target),
            ],
            stdout=self.subprocess_output(),
            stderr=self.subprocess_output(),
        )

        # Parse vdl output for relevant information
        with open(self.vdl_output_path(), encoding="utf-8") as f:
            vdl_content = f.read()
            matches = re.search(
                r'network_address:\s+"\[([0-9a-f]{1,4}:(:[0-9a-f]{1,4}){4}%qemu)\]"',
                vdl_content,
            )
            self.emu_addr = matches.group(1)

        # Create new package repo
        self.log_info("Creating package repo...")
        subprocess.check_call(
            [
                self.tool_path("pm"),
                "newrepo",
                "-repo",
                self.repo_dir(),
            ],
            stdout=self.subprocess_output(),
            stderr=self.subprocess_output(),
        )

        # Start package server
        self.log_info("Starting package server...")
        with open(
            self.package_server_log_path(), "w", encoding="utf-8"
        ) as package_server_log:
            # We want this to be a long-running process that persists after the script finishes
            # pylint: disable=consider-using-with
            self.package_server_pid = subprocess.Popen(
                [
                    self.tool_path("pm"),
                    "serve",
                    "-vt",
                    "-repo",
                    self.repo_dir(),
                    "-l",
                    ":8084",
                ],
                stdout=package_server_log,
                stderr=package_server_log,
            ).pid

        # Register package server with emulator
        self.log_info("Registering package server...")
        ssh_client = subprocess.check_output(
            [
                "ssh",
                "-i",
                self.ssh_keyfile_path(),
                "-o",
                "StrictHostKeyChecking=accept-new",
                self.emu_addr,
                "-f",
                "echo $SSH_CLIENT",
            ],
            text=True,
        )
        repo_addr = ssh_client.split()[0].replace("%", "%25")
        repo_url = f"http://[{repo_addr}]:8084/config.json"
        subprocess.check_call(
            [
                "ssh",
                "-i",
                self.ssh_keyfile_path(),
                "-o",
                "StrictHostKeyChecking=accept-new",
                self.emu_addr,
                "-f",
                f"pkgctl repo add url -f 1 -n {self.TEST_REPO_NAME} {repo_url}",
            ],
            stdout=self.subprocess_output(),
            stderr=self.subprocess_output(),
        )

        # Write to file
        self.write_to_file()

        self.log_info("Success! Your environment is ready to run tests.")

    # FIXME: shardify this
    # `facet` statement required for TCP testing via
    # protocol `fuchsia.posix.socket.Provider`. See
    # https://fuchsia.dev/fuchsia-src/development/testing/components/test_runner_framework?hl=en#legacy_non-hermetic_tests
    CML_TEMPLATE: ClassVar[
        str
    ] = """
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
    lib/{libstd_name}={rust_dir}/lib/rustlib/{rustlib_dir}/lib/{libstd_name}
    lib/{libtest_name}={rust_dir}/lib/rustlib/{rustlib_dir}/lib/{libtest_name}
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

        bin_path = os.path.abspath(args.bin_path)

        # Build a unique, deterministic name for the test using the name of the
        # binary and the last 6 hex digits of the hash of the full path
        def path_checksum(path):
            m = hashlib.sha256()
            m.update(path.encode("utf-8"))
            return m.hexdigest()[0:6]

        base_name = os.path.basename(os.path.dirname(args.bin_path))
        exe_name = base_name.lower().replace(".", "_")
        package_name = f"{exe_name}_{path_checksum(bin_path)}"

        package_dir = os.path.join(self.packages_dir(), package_name)
        cml_path = os.path.join(package_dir, "meta", f"{package_name}.cml")
        cm_path = os.path.join(package_dir, "meta", f"{package_name}.cm")
        manifest_path = os.path.join(package_dir, f"{package_name}.manifest")
        far_path = os.path.join(package_dir, f"{package_name}-0.far")

        shared_libs = args.shared_libs[: args.n]
        arguments = args.shared_libs[args.n :]

        test_output_dir = os.path.join(self.output_dir(), package_name)

        # Clean and create temporary output directory
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)

        os.mkdir(test_output_dir)

        # Open log file
        log_path = os.path.join(test_output_dir, "log")
        with open(log_path, "w", encoding="utf-8") as log_file:

            def log(msg):
                print(msg, file=log_file)
                log_file.flush()

            log(f"Bin path: {bin_path}")

            log("Setting up package...")

            # Set up package
            subprocess.check_call(
                [
                    self.tool_path("pm"),
                    "-o",
                    package_dir,
                    "-n",
                    package_name,
                    "init",
                ],
                stdout=log_file,
                stderr=log_file,
            )

            log("Writing CML...")

            # Write and compile CML
            with open(cml_path, "w", encoding="utf-8") as cml:
                # Collect environment variables
                env_vars = ""
                for var_name in self.TEST_ENV_VARS:
                    var_value = os.getenv(var_name)
                    if var_value is not None:
                        env_vars += f'\n            "{var_name}={var_value}",'

                # Default to no backtrace for test suite
                if os.getenv("RUST_BACKTRACE") == None:
                    env_vars += f'\n            "RUST_BACKTRACE=0",'

                cml.write(
                    self.CML_TEMPLATE.format(env_vars=env_vars, exe_name=exe_name)
                )

            log("Compiling CML...")

            subprocess.check_call(
                [
                    self.tool_path("cmc"),
                    "compile",
                    cml_path,
                    "--includepath",
                    ".",
                    "--output",
                    cm_path,
                ],
                stdout=log_file,
                stderr=log_file,
            )

            log("Writing manifest...")

            # Write, build, and archive manifest
            with open(manifest_path, "w", encoding="utf-8") as manifest:
                manifest.write(
                    self.MANIFEST_TEMPLATE.format(
                        bin_path=bin_path,
                        exe_name=exe_name,
                        package_dir=package_dir,
                        package_name=package_name,
                        rust_dir=self.rust_dir,
                        rustlib_dir=self.target,
                        sdk_dir=self.sdk_dir,
                        libstd_name=self.libstd_name,
                        libtest_name=self.libtest_name,
                        target_arch=self.triple_to_arch(self.target),
                    )
                )
                for shared_lib in shared_libs:
                    manifest.write(f"lib/{os.path.basename(shared_lib)}={shared_lib}\n")

            log("Compiling and archiving manifest...")

            subprocess.check_call(
                [
                    self.tool_path("pm"),
                    "-o",
                    package_dir,
                    "-m",
                    manifest_path,
                    "build",
                ],
                stdout=log_file,
                stderr=log_file,
            )
            subprocess.check_call(
                [
                    self.tool_path("pm"),
                    "-o",
                    package_dir,
                    "-m",
                    manifest_path,
                    "archive",
                ],
                stdout=log_file,
                stderr=log_file,
            )

            log("Publishing package to repo...")

            # Publish package to repo
            subprocess.check_call(
                [
                    self.tool_path("pm"),
                    "publish",
                    "-a",
                    "-repo",
                    self.repo_dir(),
                    "-f",
                    far_path,
                ],
                stdout=log_file,
                stderr=log_file,
            )

            log("Running ffx test...")

            # Run test on emulator
            subprocess.run(
                [
                    self.tool_path("ffx"),
                    "--config",
                    self.ffx_user_config_path(),
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
                stdout=log_file,
                stderr=log_file,
            )

            log("Reporting test suite output...")

            # Read test suite output
            run_summary_path = os.path.join(test_output_dir, "run_summary.json")
            if os.path.exists(run_summary_path):
                with open(run_summary_path, encoding="utf-8") as f:
                    run_summary = json.loads(f.read())

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

                if stdout_path is not None and os.path.exists(stdout_path):
                    with open(stdout_path, encoding="utf-8") as f:
                        print(f.read(), file=sys.stdout, end="")

                if stderr_path is not None and os.path.exists(stderr_path):
                    with open(stderr_path, encoding="utf-8") as f:
                        print(f.read(), file=sys.stderr, end="")
            else:
                log("Failed to open test run summary")
                return_code = 254

            log("Done!")

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
        Passing --delete-tmp will force the process to delete the files anyway.
        """

        self.log_debug("Reporting logs...")

        # Print test log files
        for test_dir in os.listdir(self.output_dir()):
            log_path = os.path.join(self.output_dir(), test_dir, "log")
            self.log_debug(f"\n---- Logs for test '{test_dir}' ----\n")
            if os.path.exists(log_path):
                with open(log_path, encoding="utf-8") as log:
                    self.log_debug(log.read())
            else:
                self.log_debug("No logs found")

        # Print the emulator log
        self.log_debug("\n---- Emulator logs ----\n")
        if os.path.exists(self.emulator_log_path()):
            with open(self.emulator_log_path(), encoding="utf-8") as log:
                self.log_debug(log.read())
        else:
            self.log_debug("No emulator logs found")

        # Print the package server log
        self.log_debug("\n---- Package server log ----\n")
        if os.path.exists(self.package_server_log_path()):
            with open(self.package_server_log_path(), encoding="utf-8") as log:
                self.log_debug(log.read())
        else:
            self.log_debug("No package server log found")

        # Print the ffx daemon log
        self.log_debug("\n---- ffx daemon log ----\n")
        if os.path.exists(self.ffx_daemon_log_path()):
            with open(self.ffx_daemon_log_path(), encoding="utf-8") as log:
                self.log_debug(log.read())
        else:
            self.log_debug("No ffx daemon log found")

        # Stop package server
        self.log_info("Stopping package server...")
        os.kill(self.package_server_pid, signal.SIGTERM)

        # Shut down the emulator
        self.log_info("Stopping emulator...")
        subprocess.check_call(
            [
                self.tool_path("fvdl"),
                "--sdk",
                "kill",
                "--launched-proto",
                self.vdl_output_path(),
            ],
            stdout=self.subprocess_output(),
            stderr=self.subprocess_output(),
        )

        # Stop ffx isolation
        self.log_info("Stopping ffx isolation...")
        self.stop_ffx_isolation()

    def delete_tmp(self):
        # Remove temporary files
        self.log_info("Deleting temporary files...")
        shutil.rmtree(self.tmp_dir(), ignore_errors=True)

    def debug(self, args):
        command = [
            self.tool_path("ffx"),
            "--config",
            self.ffx_user_config_path(),
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
                "--config",
                self.ffx_user_config_path(),
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
    return test_env.run(args)


def stop(args):
    test_env = TestEnvironment.read_from_file()
    test_env.stop()
    if not args.no_delete:
        test_env.delete_tmp()
    return 0


def delete_tmp(args):
    del args
    test_env = TestEnvironment.read_from_file()
    test_env.delete_tmp()
    return 0


def debug(args):
    test_env = TestEnvironment.read_from_file()
    test_env.debug(args)
    return 0


def syslog(args):
    test_env = TestEnvironment.read_from_file()
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
        "--rust",
        help="the directory of the installed Rust compiler for Fuchsia",
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
        "--no-delete",
        default=False,
        action="store_true",
        help="don't delete temporary files after stopping",
    )
    stop_parser.set_defaults(func=stop)

    delete_parser = subparsers.add_parser(
        "delete-tmp",
        help="deletes temporary files after the testing environment has been manually cleaned up",
    )
    delete_parser.set_defaults(func=delete_tmp)

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

    syslog_parser = subparsers.add_parser(
        "syslog", help="prints the device syslog"
    )
    syslog_parser.set_defaults(func=syslog)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
