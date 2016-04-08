# vim: ft=perl

$pdf_mode = 1;
$pdflatex = 'lualatex --shell-escape %O %S';
$out_dir = 'out';

# This improves latexmk's detection of source files and generated files.
$recorder = 1;

# Ignore always-regenerated *.pyg files from the minted package when considering
# whether to run pdflatex again.
$hash_calc_ignore_pattern{'pyg'} = '.*';
