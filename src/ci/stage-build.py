#!/usr/bin/env python3
# ignore-tidy-linelength

# Compatible with Python 3.6+

import contextlib
import getpass
import glob
import logging
import os
import pprint
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

PGO_HOST = os.environ["PGO_HOST"]

LOGGER = logging.getLogger("stage-build")

LLVM_PGO_CRATES = [
    "syn-1.0.89",
    "cargo-0.60.0",
    "serde-1.0.136",
    "ripgrep-13.0.0",
    "regex-1.5.5",
    "clap-3.1.6",
    "hyper-0.14.18"
]

RUSTC_PGO_CRATES = [
    "externs",
    "ctfe-stress-5",
    "cargo-0.60.0",
    "token-stream-stress",
    "match-stress",
    "tuple-stress",
    "diesel-1.4.8",
    "bitmaps-3.1.0"
]

LLVM_BOLT_CRATES = LLVM_PGO_CRATES


class Pipeline:
    # Paths
    def checkout_path(self) -> Path:
        """
        The root checkout, where the source is located.
        """
        raise NotImplementedError

    def downloaded_llvm_dir(self) -> Path:
        """
        Directory where the host LLVM is located.
        """
        raise NotImplementedError

    def build_root(self) -> Path:
        """
        The main directory where the build occurs.
        """
        raise NotImplementedError

    def build_artifacts(self) -> Path:
        return self.build_root() / "build" / PGO_HOST

    def rustc_stage_0(self) -> Path:
        return self.build_artifacts() / "stage0" / "bin" / "rustc"

    def cargo_stage_0(self) -> Path:
        return self.build_artifacts() / "stage0" / "bin" / "cargo"

    def rustc_stage_2(self) -> Path:
        return self.build_artifacts() / "stage2" / "bin" / "rustc"

    def opt_artifacts(self) -> Path:
        raise NotImplementedError

    def llvm_profile_dir_root(self) -> Path:
        return self.opt_artifacts() / "llvm-pgo"

    def llvm_profile_merged_file(self) -> Path:
        return self.opt_artifacts() / "llvm-pgo.profdata"

    def rustc_perf_dir(self) -> Path:
        return self.opt_artifacts() / "rustc-perf"

    def build_rustc_perf(self):
        raise NotImplementedError()

    def rustc_profile_dir_root(self) -> Path:
        return self.opt_artifacts() / "rustc-pgo"

    def rustc_profile_merged_file(self) -> Path:
        return self.opt_artifacts() / "rustc-pgo.profdata"

    def rustc_profile_template_path(self) -> Path:
        """
        The profile data is written into a single filepath that is being repeatedly merged when each
        rustc invocation ends. Empirically, this can result in some profiling data being lost. That's
        why we override the profile path to include the PID. This will produce many more profiling
        files, but the resulting profile will produce a slightly faster rustc binary.
        """
        return self.rustc_profile_dir_root() / "default_%m_%p.profraw"

    def supports_bolt(self) -> bool:
        raise NotImplementedError

    def llvm_bolt_profile_merged_file(self) -> Path:
        return self.opt_artifacts() / "bolt.profdata"


class LinuxPipeline(Pipeline):
    def checkout_path(self) -> Path:
        return Path("/checkout")

    def downloaded_llvm_dir(self) -> Path:
        return Path("/rustroot")

    def build_root(self) -> Path:
        return self.checkout_path() / "obj"

    def opt_artifacts(self) -> Path:
        return Path("/tmp/tmp-multistage/opt-artifacts")

    def build_rustc_perf(self):
        # /tmp/rustc-perf comes from the Dockerfile
        shutil.copytree("/tmp/rustc-perf", self.rustc_perf_dir())
        cmd(["chown", "-R", f"{getpass.getuser()}:", self.rustc_perf_dir()])

        with change_cwd(self.rustc_perf_dir()):
            cmd([self.cargo_stage_0(), "build", "-p", "collector"], env=dict(
                RUSTC=str(self.rustc_stage_0()),
                RUSTC_BOOTSTRAP="1"
            ))

    def supports_bolt(self) -> bool:
        return True


class WindowsPipeline(Pipeline):
    def __init__(self):
        self.checkout_dir = Path(os.getcwd())

    def checkout_path(self) -> Path:
        return self.checkout_dir

    def downloaded_llvm_dir(self) -> Path:
        return self.checkout_path() / "citools" / "clang-rust"

    def build_root(self) -> Path:
        return self.checkout_path()

    def opt_artifacts(self) -> Path:
        return self.checkout_path() / "opt-artifacts"

    def rustc_stage_0(self) -> Path:
        return super().rustc_stage_0().with_suffix(".exe")

    def cargo_stage_0(self) -> Path:
        return super().cargo_stage_0().with_suffix(".exe")

    def rustc_stage_2(self) -> Path:
        return super().rustc_stage_2().with_suffix(".exe")

    def build_rustc_perf(self):
        # rustc-perf version from 2022-07-22
        perf_commit = "3c253134664fdcba862c539d37f0de18557a9a4c"
        rustc_perf_zip_path = self.opt_artifacts() / "perf.zip"

        def download_rustc_perf():
            download_file(
                f"https://github.com/rust-lang/rustc-perf/archive/{perf_commit}.zip",
                rustc_perf_zip_path
            )
            with change_cwd(self.opt_artifacts()):
                unpack_archive(rustc_perf_zip_path)
                move_path(Path(f"rustc-perf-{perf_commit}"), self.rustc_perf_dir())
                delete_file(rustc_perf_zip_path)

        retry_action(download_rustc_perf, "Download rustc-perf")

        with change_cwd(self.rustc_perf_dir()):
            cmd([self.cargo_stage_0(), "build", "-p", "collector"], env=dict(
                RUSTC=str(self.rustc_stage_0()),
                RUSTC_BOOTSTRAP="1"
            ))

    def rustc_profile_template_path(self) -> Path:
        """
        On Windows, we don't have enough space to use separate files for each rustc invocation.
        Therefore, we use a single file for the generated profiles.
        """
        return self.rustc_profile_dir_root() / "default_%m.profraw"

    def supports_bolt(self) -> bool:
        return False


class Timer:
    def __init__(self):
        # We want this dictionary to be ordered by insertion.
        # We use `OrderedDict` for compatibility with older Python versions.
        self.stages = OrderedDict()

    @contextlib.contextmanager
    def stage(self, name: str):
        assert name not in self.stages

        start = time.time()
        exc = None
        try:
            LOGGER.info(f"Stage `{name}` starts")
            yield
        except BaseException as exception:
            exc = exception
            raise
        finally:
            end = time.time()
            duration = end - start
            self.stages[name] = duration
            if exc is None:
                LOGGER.info(f"Stage `{name}` ended: OK ({duration:.2f}s)")
            else:
                LOGGER.info(f"Stage `{name}` ended: FAIL ({duration:.2f}s)")

    def print_stats(self):
        total_duration = sum(self.stages.values())

        # 57 is the width of the whole table
        divider = "-" * 57

        with StringIO() as output:
            print(divider, file=output)
            for (name, duration) in self.stages.items():
                pct = (duration / total_duration) * 100
                name_str = f"{name}:"
                print(f"{name_str:<34} {duration:>12.2f}s ({pct:>5.2f}%)", file=output)

            total_duration_label = "Total duration:"
            print(f"{total_duration_label:<34} {total_duration:>12.2f}s", file=output)
            print(divider, file=output, end="")
            LOGGER.info(f"Timer results\n{output.getvalue()}")


@contextlib.contextmanager
def change_cwd(dir: Path):
    """
    Temporarily change working directory to `dir`.
    """
    cwd = os.getcwd()
    LOGGER.debug(f"Changing working dir from `{cwd}` to `{dir}`")
    os.chdir(dir)
    try:
        yield
    finally:
        LOGGER.debug(f"Reverting working dir to `{cwd}`")
        os.chdir(cwd)


def move_path(src: Path, dst: Path):
    LOGGER.info(f"Moving `{src}` to `{dst}`")
    shutil.move(src, dst)


def delete_file(path: Path):
    LOGGER.info(f"Deleting file `{path}`")
    os.unlink(path)


def delete_directory(path: Path):
    LOGGER.info(f"Deleting directory `{path}`")
    shutil.rmtree(path)


def unpack_archive(archive: Path):
    LOGGER.info(f"Unpacking archive `{archive}`")
    shutil.unpack_archive(archive)


def download_file(src: str, target: Path):
    LOGGER.info(f"Downloading `{src}` into `{target}`")
    urllib.request.urlretrieve(src, str(target))


def retry_action(action, name: str, max_fails: int = 5):
    LOGGER.info(f"Attempting to perform action `{name}` with retry")
    for iteration in range(max_fails):
        LOGGER.info(f"Attempt {iteration + 1}/{max_fails}")
        try:
            action()
            return
        except:
            LOGGER.error(f"Action `{name}` has failed\n{traceback.format_exc()}")

    raise Exception(f"Action `{name}` has failed after {max_fails} attempts")


def cmd(
        args: List[Union[str, Path]],
        env: Optional[Dict[str, str]] = None,
        output_path: Optional[Path] = None
):
    args = [str(arg) for arg in args]

    environment = os.environ.copy()

    cmd_str = ""
    if env is not None:
        environment.update(env)
        cmd_str += " ".join(f"{k}={v}" for (k, v) in (env or {}).items())
        cmd_str += " "
    cmd_str += " ".join(args)
    if output_path is not None:
        cmd_str += f" > {output_path}"
    LOGGER.info(f"Executing `{cmd_str}`")

    if output_path is not None:
        with open(output_path, "w") as f:
            return subprocess.run(
                args,
                env=environment,
                check=True,
                stdout=f
            )
    return subprocess.run(args, env=environment, check=True)


def run_compiler_benchmarks(
        pipeline: Pipeline,
        profiles: List[str],
        scenarios: List[str],
        crates: List[str],
        env: Optional[Dict[str, str]] = None
):
    env = env if env is not None else {}

    # Compile libcore, both in opt-level=0 and opt-level=3
    with change_cwd(pipeline.build_root()):
        cmd([
            pipeline.rustc_stage_2(),
            "--edition", "2021",
            "--crate-type", "lib",
            str(pipeline.checkout_path() / "library/core/src/lib.rs"),
            "--out-dir", pipeline.opt_artifacts()
        ], env=dict(RUSTC_BOOTSTRAP="1", **env))

        cmd([
            pipeline.rustc_stage_2(),
            "--edition", "2021",
            "--crate-type", "lib",
            "-Copt-level=3",
            str(pipeline.checkout_path() / "library/core/src/lib.rs"),
            "--out-dir", pipeline.opt_artifacts()
        ], env=dict(RUSTC_BOOTSTRAP="1", **env))

    # Run rustc-perf benchmarks
    # Benchmark using profile_local with eprintln, which essentially just means
    # don't actually benchmark -- just make sure we run rustc a bunch of times.
    with change_cwd(pipeline.rustc_perf_dir()):
        cmd([
            pipeline.cargo_stage_0(),
            "run",
            "-p", "collector", "--bin", "collector", "--",
            "profile_local", "eprintln",
            pipeline.rustc_stage_2(),
            "--id", "Test",
            "--cargo", pipeline.cargo_stage_0(),
            "--profiles", ",".join(profiles),
            "--scenarios", ",".join(scenarios),
            "--include", ",".join(crates)
        ], env=dict(
            RUST_LOG="collector=debug",
            RUSTC=str(pipeline.rustc_stage_0()),
            RUSTC_BOOTSTRAP="1",
            **env
        ))


# https://stackoverflow.com/a/31631711/1107768
def format_bytes(size: int) -> str:
    """Return the given bytes as a human friendly KiB, MiB or GiB string."""
    KB = 1024
    MB = KB ** 2  # 1,048,576
    GB = KB ** 3  # 1,073,741,824
    TB = KB ** 4  # 1,099,511,627,776

    if size < KB:
        return f"{size} B"
    elif KB <= size < MB:
        return f"{size / KB:.2f} KiB"
    elif MB <= size < GB:
        return f"{size / MB:.2f} MiB"
    elif GB <= size < TB:
        return f"{size / GB:.2f} GiB"
    else:
        return str(size)


# https://stackoverflow.com/a/63307131/1107768
def count_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def count_files_with_prefix(path: Path) -> int:
    return sum(1 for p in glob.glob(f"{path}*") if Path(p).is_file())


# https://stackoverflow.com/a/55659577/1107768
def get_path_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*"))
    return path.stat().st_size


def get_path_prefix_size(path: Path) -> int:
    """
    Get size of all files beginning with the prefix `path`.
    Alternative to shell `du -sh <path>*`.
    """
    return sum(Path(p).stat().st_size for p in glob.glob(f"{path}*"))


def get_files(directory: Path, filter: Optional[Callable[[Path], bool]] = None) -> Iterable[Path]:
    for file in os.listdir(directory):
        path = directory / file
        if filter is None or filter(path):
            yield path


def build_rustc(
        pipeline: Pipeline,
        args: List[str],
        env: Optional[Dict[str, str]] = None
):
    arguments = [
                    sys.executable,
                    pipeline.checkout_path() / "x.py",
                    "build",
                    "--target", PGO_HOST,
                    "--host", PGO_HOST,
                    "--stage", "2",
                    "library/std"
                ] + args
    cmd(arguments, env=env)


def create_pipeline() -> Pipeline:
    if sys.platform == "linux":
        return LinuxPipeline()
    elif sys.platform in ("cygwin", "win32"):
        return WindowsPipeline()
    else:
        raise Exception(f"Optimized build is not supported for platform {sys.platform}")


def gather_llvm_profiles(pipeline: Pipeline):
    LOGGER.info("Running benchmarks with PGO instrumented LLVM")
    run_compiler_benchmarks(
        pipeline,
        profiles=["Debug", "Opt"],
        scenarios=["Full"],
        crates=LLVM_PGO_CRATES
    )

    profile_path = pipeline.llvm_profile_merged_file()
    LOGGER.info(f"Merging LLVM PGO profiles to {profile_path}")
    cmd([
        pipeline.downloaded_llvm_dir() / "bin" / "llvm-profdata",
        "merge",
        "-o", profile_path,
        pipeline.llvm_profile_dir_root()
    ])

    LOGGER.info("LLVM PGO statistics")
    LOGGER.info(f"{profile_path}: {format_bytes(get_path_size(profile_path))}")
    LOGGER.info(
        f"{pipeline.llvm_profile_dir_root()}: {format_bytes(get_path_size(pipeline.llvm_profile_dir_root()))}")
    LOGGER.info(f"Profile file count: {count_files(pipeline.llvm_profile_dir_root())}")

    # We don't need the individual .profraw files now that they have been merged
    # into a final .profdata
    delete_directory(pipeline.llvm_profile_dir_root())


def gather_rustc_profiles(pipeline: Pipeline):
    LOGGER.info("Running benchmarks with PGO instrumented rustc")

    # Here we're profiling the `rustc` frontend, so we also include `Check`.
    # The benchmark set includes various stress tests that put the frontend under pressure.
    run_compiler_benchmarks(
        pipeline,
        profiles=["Check", "Debug", "Opt"],
        scenarios=["All"],
        crates=RUSTC_PGO_CRATES,
        env=dict(
            LLVM_PROFILE_FILE=str(pipeline.rustc_profile_template_path())
        )
    )

    profile_path = pipeline.rustc_profile_merged_file()
    LOGGER.info(f"Merging Rustc PGO profiles to {profile_path}")
    cmd([
        pipeline.build_artifacts() / "llvm" / "bin" / "llvm-profdata",
        "merge",
        "-o", profile_path,
        pipeline.rustc_profile_dir_root()
    ])

    LOGGER.info("Rustc PGO statistics")
    LOGGER.info(f"{profile_path}: {format_bytes(get_path_size(profile_path))}")
    LOGGER.info(
        f"{pipeline.rustc_profile_dir_root()}: {format_bytes(get_path_size(pipeline.rustc_profile_dir_root()))}")
    LOGGER.info(f"Profile file count: {count_files(pipeline.rustc_profile_dir_root())}")

    # We don't need the individual .profraw files now that they have been merged
    # into a final .profdata
    delete_directory(pipeline.rustc_profile_dir_root())


def gather_llvm_bolt_profiles(pipeline: Pipeline):
    LOGGER.info("Running benchmarks with BOLT instrumented LLVM")
    run_compiler_benchmarks(
        pipeline,
        profiles=["Check", "Debug", "Opt"],
        scenarios=["Full"],
        crates=LLVM_BOLT_CRATES
    )

    merged_profile_path = pipeline.llvm_bolt_profile_merged_file()
    profile_files_path = Path("/tmp/prof.fdata")
    LOGGER.info(f"Merging LLVM BOLT profiles to {merged_profile_path}")

    profile_files = sorted(glob.glob(f"{profile_files_path}*"))
    cmd([
        "merge-fdata",
        *profile_files,
    ], output_path=merged_profile_path)

    LOGGER.info("LLVM BOLT statistics")
    LOGGER.info(f"{merged_profile_path}: {format_bytes(get_path_size(merged_profile_path))}")
    LOGGER.info(
        f"{profile_files_path}: {format_bytes(get_path_prefix_size(profile_files_path))}")
    LOGGER.info(f"Profile file count: {count_files_with_prefix(profile_files_path)}")


def clear_llvm_files(pipeline: Pipeline):
    """
    Rustbuild currently doesn't support rebuilding LLVM when PGO options
    change (or any other llvm-related options); so just clear out the relevant
    directories ourselves.
    """
    LOGGER.info("Clearing LLVM build files")
    delete_directory(pipeline.build_artifacts() / "llvm")
    delete_directory(pipeline.build_artifacts() / "lld")


def print_binary_sizes(pipeline: Pipeline):
    bin_dir = pipeline.build_artifacts() / "stage2" / "bin"
    binaries = get_files(bin_dir)

    lib_dir = pipeline.build_artifacts() / "stage2" / "lib"
    libraries = get_files(lib_dir, lambda p: p.suffix == ".so")

    paths = sorted(binaries) + sorted(libraries)
    with StringIO() as output:
        for path in paths:
            path_str = f"{path.name}:"
            print(f"{path_str:<30}{format_bytes(path.stat().st_size):>14}", file=output)
        LOGGER.info(f"Rustc binary size\n{output.getvalue()}")


def execute_build_pipeline(timer: Timer, pipeline: Pipeline, final_build_args: List[str]):
    # Clear and prepare tmp directory
    shutil.rmtree(pipeline.opt_artifacts(), ignore_errors=True)
    os.makedirs(pipeline.opt_artifacts(), exist_ok=True)

    pipeline.build_rustc_perf()

    # Stage 1: Build rustc + PGO instrumented LLVM
    with timer.stage("Build rustc (LLVM PGO)"):
        build_rustc(pipeline, args=[
            "--llvm-profile-generate"
        ], env=dict(
            LLVM_PROFILE_DIR=str(pipeline.llvm_profile_dir_root() / "prof-%p")
        ))

    with timer.stage("Gather profiles (LLVM PGO)"):
        gather_llvm_profiles(pipeline)

    clear_llvm_files(pipeline)
    final_build_args += [
        "--llvm-profile-use",
        pipeline.llvm_profile_merged_file()
    ]

    # Stage 2: Build PGO instrumented rustc + LLVM
    with timer.stage("Build rustc (rustc PGO)"):
        build_rustc(pipeline, args=[
            "--rust-profile-generate",
            pipeline.rustc_profile_dir_root()
        ])

    with timer.stage("Gather profiles (rustc PGO)"):
        gather_rustc_profiles(pipeline)

    clear_llvm_files(pipeline)
    final_build_args += [
        "--rust-profile-use",
        pipeline.rustc_profile_merged_file()
    ]

    # Stage 3: Build rustc + BOLT instrumented LLVM
    if pipeline.supports_bolt():
        with timer.stage("Build rustc (LLVM BOLT)"):
            build_rustc(pipeline, args=[
                "--llvm-profile-use",
                pipeline.llvm_profile_merged_file(),
                "--llvm-bolt-profile-generate",
            ])
        with timer.stage("Gather profiles (LLVM BOLT)"):
            gather_llvm_bolt_profiles(pipeline)

        clear_llvm_files(pipeline)
        final_build_args += [
            "--llvm-bolt-profile-use",
            pipeline.llvm_bolt_profile_merged_file()
        ]

    # Stage 4: Build PGO optimized rustc + PGO/BOLT optimized LLVM
    with timer.stage("Final build"):
        cmd(final_build_args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s %(levelname)-4s: %(message)s",
    )

    LOGGER.info(f"Running multi-stage build using Python {sys.version}")
    LOGGER.info(f"Environment values\n{pprint.pformat(dict(os.environ), indent=2)}")

    build_args = sys.argv[1:]

    timer = Timer()
    pipeline = create_pipeline()
    try:
        execute_build_pipeline(timer, pipeline, build_args)
    except BaseException as e:
        LOGGER.error("The multi-stage build has failed")
        raise e
    finally:
        timer.print_stats()

    print_binary_sizes(pipeline)
