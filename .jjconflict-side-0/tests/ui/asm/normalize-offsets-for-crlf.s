// This file contains (some) CRLF line endings.  When codegen reports an error, the byte
// offsets into this file that it identifies require normalization or else they will not
// identify the appropriate span.  Worse still, an ICE can result if the erroneous span
// begins or ends part-way through a multibyte character such as £.
non_existent_mnemonic

// Without normalization, the three CRLF line endings below cause the diagnostic on the
// `non_existent_mnemonic` above to be spanned three bytes backward, and thus begin
// part-way inside the multibyte character in the preceding comment.
//
// NOTE: The lines of this note DELIBERATELY end with CRLF - DO NOT strip/convert them!
//       It may not be obvious if you accidentally do, eg `git diff` may appear to show
//       that the lines have been updated to the exact same content.
