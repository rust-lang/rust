fn blk1(b: block()) -> fn@() { ret fn@() { }; }

fn test1() { (blk1 {|| #debug["hi"]; })(); }

fn test2() { (blk1 {|| #debug["hi"]; }) {|| #debug["ho"]; }; }
