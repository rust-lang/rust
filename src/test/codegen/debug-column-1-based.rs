// Ensure that columns emitted in DWARF are 1-based

// ignore-windows
// compile-flags: -g -C no-prepopulate-passes

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

fn main() {      // 10
    let x = 32;  // 11
}                // 12
//  |   |
//  |   |
//  5   9        column index

// CHECK-LABEL: !DISubprogram(name: "main"
// CHECK:       !DILocalVariable(name: "x",{{.*}}line: 11{{[^0-9]}}
// CHECK-NEXT:  !DILexicalBlock({{.*}}line: 11, column: 5{{[^0-9]}}
// CHECK-NEXT:  !DILocation(line: 11, column: 9{{[^0-9]}}
// CHECK:       !DILocation(line: 12, column: 1{{[^0-9]}}
// CHECK-EMPTY:
