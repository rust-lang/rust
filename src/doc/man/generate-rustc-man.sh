#!/bin/bash
set -euo pipefail

RUSTC="${1:?Usage: $0 <path-to-rustc>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$SCRIPT_DIR/rustc.1"
INCLUDE_FILE="$(mktemp)"
trap "rm -f '$INCLUDE_FILE'" EXIT

# Parse -C help output into man format
generate_codegen_section() {
    "$RUSTC" -C help 2>&1 | awk '
    /^    -C / {
        sub(/^    -C +/, "")
        split($0, parts, / -- /)
        option = parts[1]
        desc = parts[2]
        gsub(/=val$/, "", option)
        printf ".TP\n\\fB-C %s\\fR=\\fIval\\fR\n%s\n", option, desc
    }'
}

# Build the include file with proper section ordering
generate_include_file() {
    cat <<'EOF'
[name]
rustc \- The Rust compiler

[description]
This program is a compiler for the Rust language, available at https://www.rust-lang.org.

[>options]
.SH "CODEGEN OPTIONS"
These options affect code generation. Run \fBrustc -C help\fR for the full list.

EOF

    # Dynamic codegen options
    generate_codegen_section

    cat <<'EOF'

.SH ENVIRONMENT
Some of these affect only test harness programs (generated via rustc --test);
others affect all programs which link to the Rust standard library.
.TP
\fBRUST_BACKTRACE\fR
If set to a value different than "0", produces a backtrace in the output of a program which panics.
.TP
\fBRUST_TEST_THREADS\fR
The test framework Rust provides executes tests in parallel. This variable sets the maximum number of threads used for this purpose. This setting is overridden by the --test-threads option.
.TP
\fBRUST_TEST_NOCAPTURE\fR
If set to a value other than "0", a synonym for the --nocapture flag.
.TP
\fBRUST_MIN_STACK\fR
Sets the minimum stack size for new threads.

.SH EXAMPLES
To build an executable from a source file with a main function:
    $ rustc -o hello hello.rs

To build a library from a source file:
    $ rustc --crate-type=lib hello-lib.rs

To build either with a crate (.rs) file:
    $ rustc hello.rs

To build an executable with debug info:
    $ rustc -g -o hello hello.rs

[see also]
.BR rustdoc (1)

[bugs]
See https://github.com/rust-lang/rust/issues for issues.

[author]
See https://github.com/rust-lang/rust/graphs/contributors or use `git log --all --format='%cN <%cE>' | sort -u` in the rust source distribution.

[copyright]
This work is dual-licensed under Apache 2.0 and MIT terms.
See COPYRIGHT file in the rust source distribution.
EOF
}

# Generate the include file
generate_include_file >"$INCLUDE_FILE"

# Generate man page
help2man \
    --include="$INCLUDE_FILE" \
    --no-info \
    --no-discard-stderr \
    "$RUSTC" >"$OUTPUT"

echo "Generated $OUTPUT from $RUSTC help output"
