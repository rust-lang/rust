#!/bin/sh
# Download the expected version of musl to a directory `musl`

set -eux

fname=musl-1.2.5.tar.gz
sha=a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4

mkdir musl
curl -L "https://musl.libc.org/releases/$fname" -O

case "$(uname -s)" in
    MINGW*)
        # Need to extract the second line because certutil does human output
        fsha=$(certutil -hashfile "$fname" SHA256 | sed -n '2p')
        [ "$sha" = "$fsha" ] || exit 1
    ;;
    *)
        echo "$sha  $fname" | shasum -a 256 --check || exit 1
    ;;
esac

tar -xzf "$fname" -C musl --strip-components 1
rm "$fname"
