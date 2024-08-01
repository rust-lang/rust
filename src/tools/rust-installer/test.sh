#!/usr/bin/env bash

set -e -u

if [ -x /bin/echo ]; then
    ECHO='/bin/echo'
else
    ECHO='echo'
fi

# Prints the absolute path of a directory to stdout
abs_path() {
    local path="$1"
    # Unset CDPATH because it causes havok: it makes the destination unpredictable
    # and triggers 'cd' to print the path to stdout. Route `cd`'s output to /dev/null
    # for good measure.
    (unset CDPATH && cd "$path" > /dev/null && pwd)
}

S="$(abs_path $(dirname $0))"

TEST_DIR="$S/test"
WORK_DIR="$TMP_DIR/workdir"
OUT_DIR="$TMP_DIR/outdir"
PREFIX_DIR="$TMP_DIR/prefix"

case $(uname -s) in

    MINGW* | MSYS*)
    WINDOWS=1
        ;;
esac

say() {
    echo "test: $1"
}

pre() {
    echo "test: $1"
    rm -Rf "$WORK_DIR"
    rm -Rf "$OUT_DIR"
    rm -Rf "$PREFIX_DIR"
    mkdir -p "$WORK_DIR"
    mkdir -p "$OUT_DIR"
    mkdir -p "$PREFIX_DIR"
}

need_ok() {
    if [ $? -ne 0 ]
    then
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    fi
}

fail() {
    echo
    echo "$1"
    echo
    echo "TEST FAILED!"
    echo
    exit 1
}

try() {
    set +e
    _cmd="$@"
    _output=`$@ 2>&1`
    if [ $? -ne 0 ]; then
    echo \$ "$_cmd"
    # Using /bin/echo to avoid escaping
    $ECHO "$_output"
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    else
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_CMD-}" ]; then
        echo \$ "$_cmd"
    fi
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_OUTPUT-}" ]; then
        $ECHO "$_output"
    fi
    fi
    set -e
}

expect_fail() {
    set +e
    _cmd="$@"
    _output=`$@ 2>&1`
    if [ $? -eq 0 ]; then
    echo \$ "$_cmd"
    # Using /bin/echo to avoid escaping
    $ECHO "$_output"
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    else
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_CMD-}" ]; then
        echo \$ "$_cmd"
    fi
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_OUTPUT-}" ]; then
        $ECHO "$_output"
    fi
    fi
    set -e
}

expect_output_ok() {
    set +e
    local _expected="$1"
    shift 1
    _cmd="$@"
    _output=`$@ 2>&1`
    if [ $? -ne 0 ]; then
    echo \$ "$_cmd"
    # Using /bin/echo to avoid escaping
    $ECHO "$_output"
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    elif ! echo "$_output" | grep -q "$_expected"; then
    echo \$ "$_cmd"
    $ECHO "$_output"
    echo
    echo "missing expected output '$_expected'"
    echo
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    else
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_CMD-}" ]; then
        echo \$ "$_cmd"
    fi
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_OUTPUT-}" ]; then
        $ECHO "$_output"
    fi
    fi
    set -e
}

expect_output_fail() {
    set +e
    local _expected="$1"
    shift 1
    _cmd="$@"
    _output=`$@ 2>&1`
    if [ $? -eq 0 ]; then
    echo \$ "$_cmd"
    # Using /bin/echo to avoid escaping
    $ECHO "$_output"
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    elif ! echo "$_output" | grep -q "$_expected"; then
    echo \$ "$_cmd"
    $ECHO "$_output"
    echo
    echo "missing expected output '$_expected'"
    echo
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    else
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_CMD-}" ]; then
        echo \$ "$_cmd"
    fi
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_OUTPUT-}" ]; then
        $ECHO "$_output"
    fi
    fi
    set -e
}

expect_not_output_ok() {
    set +e
    local _expected="$1"
    shift 1
    _cmd="$@"
    _output=`$@ 2>&1`
    if [ $? -ne 0 ]; then
    echo \$ "$_cmd"
    # Using /bin/echo to avoid escaping
    $ECHO "$_output"
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    elif echo "$_output" | grep -q "$_expected"; then
    echo \$ "$_cmd"
    $ECHO "$_output"
    echo
    echo "unexpected output '$_expected'"
    echo
    echo
    echo "TEST FAILED!"
    echo
    exit 1
    else
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_CMD-}" ]; then
        echo \$ "$_cmd"
    fi
    if [ -n "${VERBOSE-}" -o -n "${VERBOSE_OUTPUT-}" ]; then
        $ECHO "$_output"
    fi
    fi
    set -e
}

runtest() {
    local _testname="$1"
    if [ -n "${TESTNAME-}" ]; then
    if ! echo "$_testname" | grep -q "$TESTNAME"; then
        return 0
    fi
    fi

    pre "$_testname"
    "$_testname"
}

# Installation tests

basic_install() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
}
runtest basic_install

basic_uninstall() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/package/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/something-to-install"
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/program2"
    try test ! -e "$PREFIX_DIR/bin/bad-bin"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest basic_uninstall

not_installed_files() {
    mkdir -p "$WORK_DIR/overlay"
    touch "$WORK_DIR/overlay/not-installed"
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --non-installed-overlay="$WORK_DIR/overlay"
    try test -e "$WORK_DIR/package/not-installed"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/not-installed"
}
runtest not_installed_files

tarball_with_package_name() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc-nightly
    try "$WORK_DIR/rustc-nightly/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$OUT_DIR/rustc-nightly.tar.gz"
    try test -e "$OUT_DIR/rustc-nightly.tar.xz"
}
runtest tarball_with_package_name

install_overwrite_backup() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try mkdir -p "$PREFIX_DIR/bin"
    touch "$PREFIX_DIR/bin/program"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    # The existing program was backed up by 'install'
    try test -e "$PREFIX_DIR/bin/program.old"
}
runtest install_overwrite_backup

bulk_directory() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --bulk-dirs=dir-to-install
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR" --uninstall
    try test ! -e "$PREFIX_DIR/dir-to-install"
}
runtest bulk_directory

bulk_directory_overwrite() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --bulk-dirs=dir-to-install
    try mkdir -p "$PREFIX_DIR/dir-to-install"
    try touch "$PREFIX_DIR/dir-to-install/overwrite"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    # The file that used to exist in the directory no longer does
    try test ! -e "$PREFIX_DIR/dir-to-install/overwrite"
    # It was backed up
    try test -e "$PREFIX_DIR/dir-to-install.old/overwrite"
}
runtest bulk_directory_overwrite

bulk_directory_overwrite_existing_backup() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --bulk-dirs=dir-to-install
    try mkdir -p "$PREFIX_DIR/dir-to-install"
    try touch "$PREFIX_DIR/dir-to-install/overwrite"
    # This time we've already got an existing backup of the overwritten directory.
    # The install should still succeed.
    try mkdir -p "$PREFIX_DIR/dir-to-install~"
    try touch "$PREFIX_DIR/dir-to-install~/overwrite"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/dir-to-install/overwrite"
    try test -e "$PREFIX_DIR/dir-to-install~/overwrite"
}
runtest bulk_directory_overwrite_existing_backup

nested_bulk_directory() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --bulk-dirs=dir-to-install/qux
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/dir-to-install/qux/bar"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR" --uninstall
    try test ! -e "$PREFIX_DIR/dir-to-install/qux"
}
runtest nested_bulk_directory

only_bulk_directory_no_files() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image5" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --bulk-dirs=dir-to-install
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR" --uninstall
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
}
runtest only_bulk_directory_no_files

nested_not_installed_files() {
    mkdir -p "$WORK_DIR/overlay"
    touch "$WORK_DIR/overlay/not-installed"
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --non-installed-overlay="$WORK_DIR/overlay"
    try test -e "$WORK_DIR/package/not-installed"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/not-installed"
}
runtest nested_not_installed_files

multiple_components() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR/c1" \
    --output-dir="$OUT_DIR/c1" \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR/c2" \
    --output-dir="$OUT_DIR/c2" \
    --component-name=cargo
    try "$WORK_DIR/c1/package/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/c2/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
    try test -e "$PREFIX_DIR/bin/cargo"
    try "$WORK_DIR/c1/package/install.sh" --prefix="$PREFIX_DIR" --uninstall
    try test ! -e "$PREFIX_DIR/something-to-install"
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/program2"
    try test ! -e "$PREFIX_DIR/bin/bad-bin"
    try "$WORK_DIR/c2/package/install.sh" --prefix="$PREFIX_DIR" --uninstall
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest multiple_components

uninstall_from_installed_script() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR/c1" \
    --output-dir="$OUT_DIR/c1" \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR/c2" \
    --output-dir="$OUT_DIR/c2" \
    --component-name=cargo
    try "$WORK_DIR/c1/package/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/c2/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
    try test -e "$PREFIX_DIR/bin/cargo"
    # All components should be uninstalled by this script
    try sh "$PREFIX_DIR/lib/packagelib/uninstall.sh"
    try test ! -e "$PREFIX_DIR/something-to-install"
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/program2"
    try test ! -e "$PREFIX_DIR/bin/bad-bin"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest uninstall_from_installed_script

uninstall_from_installed_script_with_args_fails() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR/c1" \
    --output-dir="$OUT_DIR/c1" \
    --component-name=rustc
    try "$WORK_DIR/c1/package/install.sh" --prefix="$PREFIX_DIR"
    expect_output_fail "uninstall.sh does not take any arguments" \
    sh "$PREFIX_DIR/lib/packagelib/uninstall.sh" --prefix=foo
}
runtest uninstall_from_installed_script_with_args_fails

# Combined installer tests

combine_installers() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
    try test -e "$PREFIX_DIR/bin/cargo"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/something-to-install"
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/program2"
    try test ! -e "$PREFIX_DIR/bin/bad-bin"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest combine_installers

combine_three_installers() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/dir-to-install/qux/bar"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/something-to-install"
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/program2"
    try test ! -e "$PREFIX_DIR/bin/bad-bin"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
    try test ! -e "$PREFIX_DIR/dir-to-install/qux/bar"
}
runtest combine_three_installers

combine_installers_with_overlay() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    mkdir -p "$WORK_DIR/overlay"
    touch "$WORK_DIR/overlay/README"
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz" \
    --non-installed-overlay="$WORK_DIR/overlay"
    try test -e "$WORK_DIR/rust/README"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/README"
}
runtest combine_installers_with_overlay

combined_with_bulk_dirs() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc \
    --bulk-dirs=dir-to-install
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/dir-to-install"
}
runtest combined_with_bulk_dirs

combine_install_with_separate_uninstall() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc \
    --rel-manifest-dir=rustlib
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo \
    --rel-manifest-dir=rustlib
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz" \
    --rel-manifest-dir=rustlib
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/something-to-install"
    try test -e "$PREFIX_DIR/dir-to-install/foo"
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/program2"
    try test -e "$PREFIX_DIR/bin/bad-bin"
    try test -e "$PREFIX_DIR/bin/cargo"
    try "$WORK_DIR/rustc/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/something-to-install"
    try test ! -e "$PREFIX_DIR/dir-to-install/foo"
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/program2"
    try test ! -e "$PREFIX_DIR/bin/bad-bin"
    try "$WORK_DIR/cargo/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest combine_install_with_separate_uninstall

select_components_to_install() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --components=rustc
    try test -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --components=cargo
    try test ! -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --components=rust-docs
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --components=rustc,cargo
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" \
    --components=rustc,cargo,rust-docs
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest select_components_to_install

select_components_to_uninstall() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --components=rustc
    try test ! -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --components=cargo
    try test -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --components=rust-docs
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --components=rustc,cargo
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" \
    --components=rustc,cargo,rust-docs
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try test ! -e "$PREFIX_DIR/lib/packagelib"
}
runtest select_components_to_uninstall

invalid_component() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    expect_output_fail "unknown component" "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" \
    --components=foo
}
runtest invalid_component

without_components() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --without=rust-docs
    try test -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --without=rust-docs,cargo
    try test -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --without=rust-docs,rustc
    try test ! -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test ! -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR"
}
runtest without_components

# --uninstall --without is kind of weird,
# --without causes components to remain installed
uninstall_without_components() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --without=rust-docs
    try test ! -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --without=rust-docs,cargo
    try test ! -e "$PREFIX_DIR/bin/program"
    try test -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    try "$WORK_DIR/rust/install.sh --uninstall" --prefix="$PREFIX_DIR" --without=rust-docs,rustc
    try test -e "$PREFIX_DIR/bin/program"
    try test ! -e "$PREFIX_DIR/bin/cargo"
    try test -e "$PREFIX_DIR/baz"
}
runtest uninstall_without_components

without_any_components() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    expect_output_fail "no components selected for installation" \
    "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --without=rust-docs,rustc,cargo
}
runtest without_any_components

uninstall_without_any_components() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR"
    expect_output_fail "no components selected for uninstallation" \
    "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" \
    --uninstall --without=rust-docs,rustc,cargo
}
runtest uninstall_without_any_components

list_components() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    expect_output_ok "rustc" "$WORK_DIR/rust/install.sh" --list-components
    expect_output_ok "cargo" "$WORK_DIR/rust/install.sh" --list-components
    expect_output_ok "rust-docs" "$WORK_DIR/rust/install.sh" --list-components
}
runtest list_components

combined_remains() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rustc \
    --component-name=rustc
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image3" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=cargo \
    --component-name=cargo
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image4" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust-docs \
    --component-name=rust-docs
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz,$OUT_DIR/rust-docs.tar.gz"
    for component in rustc cargo rust-docs; do
    # bootstrap wants the original extracted package intact too
    try test -d "$WORK_DIR/$component/$component"
    try test -d "$WORK_DIR/rust/$component"
    done
}
runtest combined_remains

# Smoke tests

cannot_write_error() {
    # chmod doesn't work on windows
    if [ ! -n "${WINDOWS-}" ]; then
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR"
    chmod u-w "$PREFIX_DIR"
    expect_fail "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    chmod u+w "$PREFIX_DIR"
    fi
}
runtest cannot_write_error

cannot_install_to_installer() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=my-package
    expect_output_fail "cannot install to same directory as installer" \
    "$WORK_DIR/my-package/install.sh" --prefix="$WORK_DIR/my-package"
}
runtest cannot_install_to_installer

upgrade_from_future_installer_error() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --rel-manifest-dir=rustlib
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    echo 100 > "$PREFIX_DIR/lib/rustlib/rust-installer-version"
    expect_fail "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
}
runtest upgrade_from_future_installer_error

destdir() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --destdir="$PREFIX_DIR/" --prefix=prefix
    try test -e "$PREFIX_DIR/prefix/bin/program"
}
runtest destdir

destdir_no_trailing_slash() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --destdir="$PREFIX_DIR" --prefix=prefix
    try test -e "$PREFIX_DIR/prefix/bin/program"
}
runtest destdir_no_trailing_slash

disable_verify_noop() {
    # Obsolete --disable-verify flag doesn't generate error
    try sh "$S/gen-installer.sh" \
       --image-dir="$TEST_DIR/image1" \
       --work-dir="$WORK_DIR" \
       --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR" --disable-verify
}
runtest disable_verify_noop

create_log() {
    try sh "$S/gen-installer.sh" \
       --image-dir="$TEST_DIR/image1" \
       --work-dir="$WORK_DIR" \
       --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/lib/packagelib/install.log"
    local _log="$(cat "$PREFIX_DIR/lib/packagelib/install.log")"
    if [ -z "$_log" ]; then
    fail "log is empty"
    fi
}
runtest create_log

leave_log_after_failure() {
    # chmod doesn't work on windows
    if [ ! -n "${WINDOWS-}" ]; then
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR"
    mkdir -p "$PREFIX_DIR/lib/packagelib"
    touch "$PREFIX_DIR/lib/packagelib/components"
    chmod u-w "$PREFIX_DIR/lib/packagelib/components"
    expect_fail "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    chmod u+w "$PREFIX_DIR/lib/packagelib/components"
    try test -e "$PREFIX_DIR/lib/packagelib/install.log"
    local _log="$(cat "$PREFIX_DIR/lib/packagelib/install.log")"
    if [ -z "$_log" ]; then
        fail "log is empty"
    fi
    # script should tell user where the logs are
    if ! grep -q "see logs at" "$PREFIX_DIR/lib/packagelib/install.log"; then
        fail "missing log message"
    fi
    fi
}
runtest leave_log_after_failure

# https://github.com/rust-lang/rust-installer/issues/22
help() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --help
}
runtest help

# https://github.com/rust-lang/rust-installer/issues/31
CDPATH_does_not_destroy_things() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    cd "$WORK_DIR" || exit 1
    export CDPATH="../$(basename $WORK_DIR)/foo"
    try sh "package/install.sh" --prefix="$PREFIX_DIR"
    cd "$S" || exit 1
    cd "$PREFIX_DIR" || exit 1
    export CDPATH="../$(basename $PREFIX_DIR)"
    try sh "lib/packagelib/uninstall.sh"
    cd "$S" || exit 1
    unset CDPATH
}
runtest CDPATH_does_not_destroy_things

docdir_default() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image-docdir1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR"
    try test -e "$PREFIX_DIR/share/doc/rust/README"
    try test -e "$PREFIX_DIR/share/doc/rust/rustdocs.txt"
}
runtest docdir_default

docdir() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image-docdir1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR"
    try mkdir "$WORK_DIR/docdir"
    try "$WORK_DIR/package/install.sh" --prefix="$PREFIX_DIR" --docdir="$WORK_DIR/docdir"
    try test -e "$WORK_DIR/docdir/README"
    try test -e "$WORK_DIR/docdir/rustdocs.txt"
}
runtest docdir

docdir_combined() {
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image-docdir1" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
        --package-name="rustc" \
        --component-name="rustc"
    try sh "$S/gen-installer.sh" \
    --image-dir="$TEST_DIR/image-docdir2" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
        --package-name="cargo" \
        --component-name="cargo"
    try sh "$S/combine-installers.sh" \
    --work-dir="$WORK_DIR" \
    --output-dir="$OUT_DIR" \
    --package-name=rust \
    --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz"
    try mkdir "$WORK_DIR/docdir"
    try "$WORK_DIR/rust/install.sh" --prefix="$PREFIX_DIR" --docdir="$WORK_DIR/docdir"
    try test -e "$WORK_DIR/docdir/README"
    try test -e "$WORK_DIR/docdir/rustdocs.txt"
    try test -e "$WORK_DIR/docdir/README"
    try test -e "$WORK_DIR/docdir/cargodocs.txt"
}
runtest docdir_combined

combine_installers_different_input_compression_formats() {
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rustc \
        --component-name=rustc \
        --compression-formats=xz
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image3" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=cargo \
        --component-name=cargo \
        --compression-formats=gz
    try sh "$S/combine-installers.sh" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rust \
        --input-tarballs="$OUT_DIR/rustc.tar.xz,$OUT_DIR/cargo.tar.gz"

    try test -e "${OUT_DIR}/rust.tar.gz"
    try test -e "${OUT_DIR}/rust.tar.xz"
}
runtest combine_installers_different_input_compression_formats

generate_compression_formats_one() {
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name="rustc" \
        --component-name="rustc" \
        --compression-formats="xz"

    try test ! -e "${OUT_DIR}/rustc.tar.gz"
    try test -e "${OUT_DIR}/rustc.tar.xz"
}
runtest generate_compression_formats_one

generate_compression_formats_multiple() {
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name="rustc" \
        --component-name="rustc" \
        --compression-formats="gz,xz"

    try test -e "${OUT_DIR}/rustc.tar.gz"
    try test -e "${OUT_DIR}/rustc.tar.xz"
}
runtest generate_compression_formats_multiple

generate_compression_formats_error() {
    expect_fail sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name="rustc" \
        --component-name="rustc" \
        --compression-formats="xz,foobar"
}
runtest generate_compression_formats_error

combine_compression_formats_one() {
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rustc \
        --component-name=rustc
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image3" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=cargo \
        --component-name=cargo
    try sh "$S/combine-installers.sh" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rust \
        --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz" \
        --compression-formats=xz

    try test ! -e "${OUT_DIR}/rust.tar.gz"
    try test -e "${OUT_DIR}/rust.tar.xz"
}
runtest combine_compression_formats_one

combine_compression_formats_multiple() {
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rustc \
        --component-name=rustc
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image3" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=cargo \
        --component-name=cargo
    try sh "$S/combine-installers.sh" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rust \
        --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz" \
        --compression-formats=xz,gz

    try test -e "${OUT_DIR}/rust.tar.gz"
    try test -e "${OUT_DIR}/rust.tar.xz"
}
runtest combine_compression_formats_multiple

combine_compression_formats_error() {
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image1" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rustc \
        --component-name=rustc
    try sh "$S/gen-installer.sh" \
        --image-dir="$TEST_DIR/image3" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=cargo \
        --component-name=cargo
    expect_fail sh "$S/combine-installers.sh" \
        --work-dir="$WORK_DIR" \
        --output-dir="$OUT_DIR" \
        --package-name=rust \
        --input-tarballs="$OUT_DIR/rustc.tar.gz,$OUT_DIR/cargo.tar.gz" \
        --compression-formats=xz,foobar
}
runtest combine_compression_formats_error

tarball_compression_formats_one() {
    try cp -r "${TEST_DIR}/image1" "${WORK_DIR}/image"
    try sh "$S/make-tarballs.sh" \
        --input="${WORK_DIR}/image" \
        --work-dir="${WORK_DIR}" \
        --output="${OUT_DIR}/rustc" \
        --compression-formats="xz"

    try test ! -e "${OUT_DIR}/rustc.tar.gz"
    try test -e "${OUT_DIR}/rustc.tar.xz"
}
runtest tarball_compression_formats_one

tarball_compression_formats_multiple() {
    try cp -r "${TEST_DIR}/image1" "${WORK_DIR}/image"
    try sh "$S/make-tarballs.sh" \
        --input="${WORK_DIR}/image" \
        --work-dir="${WORK_DIR}" \
        --output="${OUT_DIR}/rustc" \
        --compression-formats="xz,gz"

    try test -e "${OUT_DIR}/rustc.tar.gz"
    try test -e "${OUT_DIR}/rustc.tar.xz"
}
runtest tarball_compression_formats_multiple

tarball_compression_formats_error() {
    try cp -r "${TEST_DIR}/image1" "${WORK_DIR}/image"
    expect_fail sh "$S/make-tarballs.sh" \
        --input="${WORK_DIR}/image" \
        --work-dir="${WORK_DIR}" \
        --output="${OUT_DIR}/rustc" \
        --compression-formats="xz,foobar"
}
runtest tarball_compression_formats_error

echo
echo "TOTAL SUCCESS!"
echo
