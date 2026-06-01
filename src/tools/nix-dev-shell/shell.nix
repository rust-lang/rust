{
  pkgs ? import <nixpkgs> { },
}:
let
  inherit (pkgs.lib) lists attrsets makeLibraryPath;

  x = pkgs.callPackage ./x { };
  inherit (x.passthru) cacert env;
in
pkgs.mkShell {
  name = "rustc-shell";

  inputsFrom = [ x ];
  packages = [
    pkgs.git
    pkgs.nix
    pkgs.glibc.out
    pkgs.glibc.static
    x
    # Get the runtime deps of the x wrapper
  ] ++ lists.flatten (attrsets.attrValues env);

  env = {
    # Avoid creating text files for ICEs.
    RUSTC_ICE = 0;
    SSL_CERT_FILE = cacert;
    # cargo seems to dlopen libcurl, so we need it in the ld library path
    LD_LIBRARY_PATH = "${makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.curl]}";
  };
}
