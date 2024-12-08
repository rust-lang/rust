{
  description = "rustc dev shell";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
	flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        x = import ./x { inherit pkgs; };
      in
      {
        devShells.default = with pkgs; mkShell {
          name = "rustc-dev-shell";
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
        };
      }
    );
}
