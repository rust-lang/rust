{
  self,
  pkgs,
  lib,
  stdenv,
  rustc,
  python3,
  makeBinaryWrapper,
  # Bootstrap
  curl,
  pkg-config,
  libiconv,
  openssl,
  patchelf,
  cacert,
  zlib,
  # LLVM Deps
  ninja,
  cmake,
  glibc
}:
stdenv.mkDerivation {
  strictDeps = true;
  pname = "x";
  version = "none";

  outputs = [
    "out"
    "unwrapped"
  ];

  src = ./x.rs;
  dontUnpack = true;

  nativeBuildInputs = [rustc makeBinaryWrapper];

  env.PYTHON = python3.interpreter;
  buildPhase = ''
    rustc -Copt-level=3 --crate-name x $src --out-dir $unwrapped/bin
  '';

  installPhase = ''
    makeWrapper $unwrapped/bin/x $out/bin/x \
      --set-default SSL_CERT_FILE "${cacert}/etc/ssl/certs/ca-bundle.crt" \
      --prefix CPATH ";" "${lib.makeSearchPath "include" [libiconv]}" \
      --prefix PATH : ${lib.makeBinPath [python3 patchelf curl pkg-config cmake ninja]} \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [openssl glibc.static zlib stdenv.cc.cc.lib]}
    '';

  meta = {
    description = "Helper for rust-lang/rust x.py";
    homepage = "https://github.com/rust-lang/rust/blob/master/src/tools/x";
    license = lib.licenses.mit;
    mainProgram = "x";
  };
}
