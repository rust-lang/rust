{
  pkgs ? import <nixpkgs> { },
}:
pkgs.stdenv.mkDerivation {
  name = "x";

  src = ./x.rs;
  dontUnpack = true;

  nativeBuildInputs = with pkgs; [ rustc ];

  buildPhase = ''
    PYTHON=${pkgs.lib.getExe pkgs.python3} rustc -Copt-level=3 --crate-name x $src --out-dir $out/bin
  '';

  meta = with pkgs.lib; {
    description = "Helper for rust-lang/rust x.py";
    homepage = "https://github.com/rust-lang/rust/blob/master/src/tools/x";
    license = licenses.mit;
    mainProgram = "x";
  };
}
