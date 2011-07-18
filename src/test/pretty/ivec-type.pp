// pp-exact:ivec-type.pp

fn f1(x: [int]) { }

fn g1() { f1(~[1, 2, 3]); }

fn f2(x: [int]) { }

fn g2() { f2(~[1, 2, 3]); }
