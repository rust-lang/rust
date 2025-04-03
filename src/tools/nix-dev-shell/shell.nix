{pkgs ? import <nixpkgs> {}}: let
  x = pkgs.callPackage ./x {};
in
  pkgs.mkShell {
    name = "rustc-shell";

    packages = [
      pkgs.git
      pkgs.nix
      pkgs.glibc
      x
    ];

    env = {
      # Avoid creating text files for ICEs.
      RUSTC_ICE = 0;
    };
  }
