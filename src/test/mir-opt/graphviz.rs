// Test graphviz output
// compile-flags: -Z dump-mir-graphviz

// ignore-tidy-linelength

fn main() {}

// END RUST SOURCE
// START rustc.main.mir_map.0.dot
// digraph Mir_0_12 { // The name here MUST be an ASCII identifier.
//     graph [fontname="monospace"];
//     node [fontname="monospace"];
//     edge [fontname="monospace"];
//     label=<fn main() -&gt; ()<br align="left"/>>;
//     bb0 [shape="none", label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="gray" align="center" colspan="1">0</td></tr><tr><td align="left" balign="left">_0 = ()<br/></td></tr><tr><td align="left">goto</td></tr></table>
// >];
//     bb1 [shape="none", label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="gray" align="center" colspan="1">1</td></tr><tr><td align="left">resume</td></tr></table>
// >];
//     bb2 [shape="none", label=<<table border="0" cellborder="1" cellspacing="0"><tr><td bgcolor="gray" align="center" colspan="1">2</td></tr><tr><td align="left">return</td></tr></table>
// >];
//     bb0 -> bb2 [label=""];
// }
// END rustc.main.mir_map.0.dot
