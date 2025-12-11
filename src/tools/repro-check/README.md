# repro-check

repro-check is a lightweight tool designed to verify the reproducibility of Rust compiler builds.

It works by creating two separate copies of the Rust source tree, building stage-2 (or a full distribution)
in each copy, and then comparing the resulting sysroots using SHA-256 checksums.
If any discrepancies are detected, repro-check generates a detailed HTML report highlighting the differences
and exits with a non-zero status.

This tool is ideal for developers and CI systems aiming to ensure deterministic, reproducible builds of Rust.

## How to build and run:

```bash
./x.py build src/tools/repro-check
./build/<your-host-triple>/stage1-tools-bin/repro-check --help

# simplest run (host target, stage 2 only)
./build/x86_64-unknown-linux-gnu/stage1-tools-bin/repro-check

# Recommended usage
./build/x86_64-unknown-linux-gnu/stage1-tools-bin/repro-check \
    --jobs 16 \
    --exclude-pattern .so \
    --path-delta 10 \
    --html-output my-report.html

# full distribution (takes a long time)
./build/x86_64-unknown-linux-gnu/stage1-tools-bin/repro-check --full-dist

# start from scratch
./build/x86_64-unknown-linux-gnu/stage1-tools-bin/repro-check --clean

All flags:

--src-root <dir>         path to the rust checkout (default: current dir)
--target <triple>        build for this target (default: host)
--jobs <n>               parallel jobs (default: number of CPUs)
--html-output <file>     report file (default: repro_report.html)
--exclude-pattern <pat>  ignore files ending with <pat> (can be repeated)
--path-delta <n>         make the second build’s path longer by n segments (default 10)
--full-dist              build a complete distribution instead of stage 2
--clean                  delete the workspace first
--skip-copy              reuse an existing workspace (don’t copy source)
--verbose                print more details

- Tests: Run `cargo test` in src/tools/repro-check.
