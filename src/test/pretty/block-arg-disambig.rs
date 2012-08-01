// FIXME: The disambiguation the pretty printer does here
// is probably not necessary anymore (#2882)

fn blk1(b: fn()) -> fn@() { ret fn@() { }; }
fn test1() { (do blk1 { debug!{"hi"}; })(); }
