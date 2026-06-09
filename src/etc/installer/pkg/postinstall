#!/bin/sh

source_dir="$(dirname "$0")"
dest_dir="$2"
package_id="$INSTALL_PKG_SESSION_ID"

if [ -z "$source_dir" ]; then
    exit 1
fi
if [ -z "$dest_dir" ]; then
    exit 1
fi
if [ -z "$package_id" ]; then
    exit 1
fi

if [ "$package_id" = "org.rust-lang.uninstall" ]; then
    if [ ! -e "$dest_dir/lib/rustlib/uninstall.sh" ]; then
	exit 1
    fi
    sh "$dest_dir/lib/rustlib/uninstall.sh"
else
    sh "$source_dir/install.sh" --prefix="$dest_dir"
fi

exit 0
