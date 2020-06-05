// checks case typos with libstd::str structs
fn main(){}

fn test_bytes(_x: bytes){}
//~^ ERROR: cannot find type `bytes` in this scope
fn test_charind(_x: Charindices){}
//~^ ERROR: cannot find type `Charindices` in this scope
fn test_chars(_x: chars){}
//~^ ERROR: cannot find type `chars` in this scope
fn test_encutf16(_x: EncodeUTF16){}
//~^ ERROR: cannot find type `EncodeUTF16` in this scope
fn test_escdflt(_x: Escapedefault){}
//~^ ERROR: cannot find type `Escapedefault` in this scope
fn test_escuni(_x: Escapeunicode){}
//~^ ERROR: cannot find type `Escapeunicode` in this scope
fn test_lines(_x: lines){}
//~^ ERROR: cannot find type `lines` in this scope
fn test_matchind(_x: Matchindices){}
//~^ ERROR: cannot find type `Matchindices` in this scope
fn test_rmatchind(_x: RmatchIndices){}
//~^ ERROR: cannot find type `RmatchIndices` in this scope
fn test_rmatch(_x: Rmatches){}
//~^ ERROR: cannot find type `Rmatches` in this scope
fn test_rsplit(_x: Rsplit){}
//~^ ERROR: cannot find type `Rsplit` in this scope
fn test_rsplitn(_x: RSplitn){}
//~^ ERROR: cannot find type `RSplitn` in this scope
fn test_rsplitterm(_x: RsplitTerminator){}
//~^ ERROR: cannot find type `RsplitTerminator` in this scope
fn test_split(_x: split){}
//~^ ERROR: cannot find type `split` in this scope
fn test_splitasciiws(_x: SplitASCIIWhitespace){}
//~^ ERROR: cannot find type `SplitASCIIWhitespace` in this scope
fn test_splitn(_x: Splitn){}
//~^ ERROR: cannot find type `Splitn` in this scope
fn test_splitterm(_x: Splitterminator){}
//~^ ERROR: cannot find type `Splitterminator` in this scope
fn test_splitws(_x: Splitwhitespace){}
//~^ ERROR: cannot find type `Splitwhitespace` in this scope
