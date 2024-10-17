{ pkgs ? import <nixpkgs> {} }:
let 
  x = import ./x { inherit pkgs; };
in
pkgs.mkShell {
  name = "rustc";
  nativeBuildInputs = with pkgs; [
    binutils cmake ninja pkg-config python3 git curl cacert patchelf nix
  ];
  buildInputs = with pkgs; [
    openssl glibc.out glibc.static x
  ];
  # Avoid creating text files for ICEs.
  RUSTC_ICE = "0";
  # Provide `libstdc++.so.6` for the self-contained lld.
  LD_LIBRARY_PATH = "${with pkgs; lib.makeLibraryPath [
    stdenv.cc.cc.lib
  ]}";
}
