#!/bin/sh
#
# This is a script that can be used in each book's CI to validate links using
# the same tool as rust-lang/rust.
#
# This requires the rust-docs rustup component to be installed in the nightly
# toolchain.
#
# Usage:
#   ./linkcheck.sh <name-of-book>
#
# Options:
#
# -i        "Iterative" mode. The script will not clean up after it is done so
#           you can inspect the result, and re-run more quickly.
#
# --all     Check all books. This can help make sure you don't break links
#           from other books into your book.

set -e

if [ ! -f book.toml ] && [ ! -f src/SUMMARY.md ]
then
    echo "Run command in root directory of the book."
    exit 1
fi

html_dir="$(rustc +nightly --print sysroot)/share/doc/rust/html"

if [ ! -d "$html_dir" ]
then
    echo "HTML docs are missing from sysroot: $html_dir"
    echo "Make sure the nightly rust-docs rustup component is installed."
    exit 1
fi

# Avoid failure caused by newer mdbook.
export MDBOOK_OUTPUT__HTML__INPUT_404=""

book_name=""
# Iterative will avoid cleaning up, so you can quickly run it repeatedly.
iterative=0
# If "1", test all books, else only this book.
all_books=0

while [ "$1" != "" ]
do
    case "$1" in
        -i)
            iterative=1
            ;;
        --all)
            all_books=1
            ;;
        *)
            if [ -n "$book_name" ]
            then
                echo "only one argument allowed"
                exit 1
            fi
            book_name="$1"
            ;;
    esac
    shift
done

if [ -z "$book_name" ]
then
    echo "usage: $0 <name-of-book>"
    exit 1
fi

if [ ! -d "$html_dir/$book_name" ]
then
    echo "book name \"$book_name\" not found in sysroot \"$html_dir\""
    exit 1
fi

if [ "$iterative" = "0" ]
then
    echo "Cleaning old directories..."
    rm -rf linkcheck linkchecker
fi

if [ ! -e "linkchecker/main.rs" ] || [ "$iterative" = "0" ]
then
    echo "Downloading linkchecker source..."
    mkdir linkchecker
    curl -o linkchecker/Cargo.toml \
        https://raw.githubusercontent.com/rust-lang/rust/master/src/tools/linkchecker/Cargo.toml
    curl -o linkchecker/main.rs \
        https://raw.githubusercontent.com/rust-lang/rust/master/src/tools/linkchecker/main.rs
fi

echo "Building book \"$book_name\"..."
mdbook build

cp -R "$html_dir" linkcheck
rm -rf "linkcheck/$book_name"
cp -R book "linkcheck/$book_name"

if [ "$all_books" = "1" ]
then
    check_path="linkcheck"
else
    check_path="linkcheck/$book_name"
fi
echo "Running linkchecker on \"$check_path\"..."
cargo run --manifest-path=linkchecker/Cargo.toml -- "$check_path"

if [ "$iterative" = "0" ]
then
    rm -rf linkcheck linkchecker
fi

echo "Link check completed successfully!"
