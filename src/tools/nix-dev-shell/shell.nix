{
  pkgs ? import <nixpkgs> { },
}:
let
  inherit (pkgs.lib) lists attrsets;

  x = pkgs.callPackage ./x { };
  inherit (x.passthru) cacert env;
in
pkgs.mkShell {
  name = "rustc-shell";

  inputsFrom = [ x ];
  packages = [
    pkgs.git
    pkgs.nix
    x
    # Get the runtime deps of the x wrapper
  ] ++ lists.flatten (attrsets.attrValues env);

  env = {
    # Avoid creating text files for ICEs.
    RUSTC_ICE = 0;
    SSL_CERT_FILE = cacert;
  };
}
