{
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
  glibc,
}:
stdenv.mkDerivation (self: {
  strictDeps = true;
  name = "x-none";

  outputs = [
    "out"
    "unwrapped"
  ];

  src = ./x.rs;
  dontUnpack = true;

  nativeBuildInputs = [
    rustc
    makeBinaryWrapper
  ];

  env.PYTHON = python3.interpreter;
  buildPhase = ''
    rustc -Copt-level=3 --crate-name x $src --out-dir $unwrapped/bin
  '';

  installPhase =
    let
      inherit (self.passthru) cacert env;
    in
    ''
      makeWrapper $unwrapped/bin/x $out/bin/x \
        --set-default SSL_CERT_FILE ${cacert} \
        --prefix CPATH ";" "${lib.makeSearchPath "include" env.cpath}" \
        --prefix PATH : ${lib.makeBinPath env.path} \
        --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath env.ldLib}
    '';

  # For accessing them in the devshell
  passthru = {
    env = {
      cpath = [ libiconv ];
      path = [
        python3
        patchelf
        curl
        pkg-config
        cmake
        ninja
        stdenv.cc
      ];
      ldLib = [
        openssl
        zlib
        stdenv.cc.cc.lib
      ];
    };
    cacert = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  };

  meta = {
    description = "Helper for rust-lang/rust x.py";
    homepage = "https://github.com/rust-lang/rust/blob/master/src/tools/x";
    license = lib.licenses.mit;
    mainProgram = "x";
  };
})
