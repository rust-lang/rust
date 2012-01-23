fn blk1(b: fn()) -> fn@() { ret fn@() { }; }
fn test1() { (blk1 {|| #debug["hi"]; })(); }
