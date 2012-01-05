fn blk1(b: block()) -> fn@() { ret fn@() { }; }
fn test1() { (blk1 {|| #debug["hi"]; })(); }
