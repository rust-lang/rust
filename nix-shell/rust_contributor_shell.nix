# Rust Contributor Shell
let
  # Unstable Channel | Rolling Release
  pkgs = import (fetchTarball("channel:nixpkgs-unstable")) { };

  packages = with pkgs; [
    pkg-config
    rustc
    cargo
    rustfmt
    rust-analyzer
    python3
    git
    libgcc
    gnumake
    curl
    cmake
    libstdcxx5
  ];
in
pkgs.mkShell {
  buildInputs = packages;
}
