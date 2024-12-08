#!/bin/bash
# ignore-tidy-linelength

set -eux

arch=$1
binutils_version=2.40
freebsd_version=12.3
triple=$arch-unknown-freebsd12
sysroot=/usr/local/$triple

hide_output() {
  set +x
  local on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  local ping_loop_pid=$!
  "$@" &> /tmp/build.log
  trap - ERR
  kill $ping_loop_pid
  set -x
}

# First up, build binutils
mkdir binutils
cd binutils
curl https://ftp.gnu.org/gnu/binutils/binutils-${binutils_version}.tar.bz2 | tar xjf -
mkdir binutils-build
cd binutils-build
hide_output ../binutils-${binutils_version}/configure \
  --target="$triple" --with-sysroot="$sysroot"
hide_output make -j"$(getconf _NPROCESSORS_ONLN)"
hide_output make install
cd ../..
rm -rf binutils

# Next, download the FreeBSD libraries and header files
mkdir -p "$sysroot"
case $arch in
  (x86_64) freebsd_arch=amd64 ;;
  (i686) freebsd_arch=i386 ;;
esac

files_to_extract=(
"./usr/include"
"./usr/lib/*crt*.o"
)
# Try to unpack only the libraries the build needs, to save space.
for lib in c cxxrt gcc_s m thr util; do
  files_to_extract=("${files_to_extract[@]}" "./lib/lib${lib}.*" "./usr/lib/lib${lib}.*")
done
for lib in c++ c_nonshared compiler_rt execinfo gcc pthread rt ssp_nonshared procstat devstat kvm memstat; do
  files_to_extract=("${files_to_extract[@]}" "./usr/lib/lib${lib}.*")
done

# Originally downloaded from:
# URL=https://download.freebsd.org/ftp/releases/${freebsd_arch}/${freebsd_version}-RELEASE/base.txz
URL=https://ci-mirrors.rust-lang.org/rustc/2022-05-06-freebsd-${freebsd_version}-${freebsd_arch}-base.txz
curl "$URL" | tar xJf - -C "$sysroot" --wildcards "${files_to_extract[@]}"

# Clang can do cross-builds out of the box, if we give it the right
# flags.  (The local binutils seem to work, but they set the ELF
# header "OS/ABI" (EI_OSABI) field to SysV rather than FreeBSD, so
# there might be other problems.)
#
# The --target option is last because the cross-build of LLVM uses
# --target without an OS version ("-freebsd" vs. "-freebsd12").  This
# makes Clang default to libstdc++ (which no longer exists), and also
# controls other features, like GNU-style symbol table hashing and
# anything predicated on the version number in the __FreeBSD__
# preprocessor macro.
for tool in clang clang++; do
  tool_path=/usr/local/bin/${triple}-${tool}
  cat > "$tool_path" <<EOF
#!/bin/sh
exec $tool --sysroot=$sysroot --prefix=${sysroot}/bin "\$@" --target=$triple
EOF
  chmod +x "$tool_path"
done
