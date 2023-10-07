// Verifies that integer types are normalized.
//
// needs-sanitizer-cfi
// compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-normalize-integers -Copt-level=0

#![crate_type="lib"]

extern crate core;
use core::ffi::*;

pub fn foo0(_: bool) { }
// CHECK: define{{.*}}foo0{{.*}}!type ![[TYPE0:[0-9]+]] !type !{{[0-9]+}}
pub fn foo1(_: bool, _: c_uchar) { }
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}}
pub fn foo2(_: bool, _: c_uchar, _: c_uchar) { }
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}}
pub fn foo3(_: isize) { }
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}}
pub fn foo4(_: isize, _: c_long) { }
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}}
pub fn foo5(_: isize, _: c_long, _: c_longlong) { }
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}}
pub fn foo6(_: usize) { }
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}}
pub fn foo7(_: usize, _: c_ulong) { }
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}}
pub fn foo8(_: usize, _: c_ulong, _: c_ulonglong) { }
// CHECK: define{{.*}}foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}}
pub fn foo9(_: c_schar) { }
// CHECK: define{{.*}}foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}}
pub fn foo10(_: c_char, _: c_schar) { }
// CHECK: define{{.*}}foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}}
pub fn foo11(_: c_char, _: c_schar, _: c_schar) { }
// CHECK: define{{.*}}foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}}
pub fn foo12(_: c_int) { }
// CHECK: define{{.*}}foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}}
pub fn foo13(_: c_int, _: c_int) { }
// CHECK: define{{.*}}foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}}
pub fn foo14(_: c_int, _: c_int, _: c_int) { }
// CHECK: define{{.*}}foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}}
pub fn foo15(_: c_short) { }
// CHECK: define{{.*}}foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}}
pub fn foo16(_: c_short, _: c_short) { }
// CHECK: define{{.*}}foo16{{.*}}!type ![[TYPE16:[0-9]+]] !type !{{[0-9]+}}
pub fn foo17(_: c_short, _: c_short, _: c_short) { }
// CHECK: define{{.*}}foo17{{.*}}!type ![[TYPE17:[0-9]+]] !type !{{[0-9]+}}
pub fn foo18(_: c_uint) { }
// CHECK: define{{.*}}foo18{{.*}}!type ![[TYPE18:[0-9]+]] !type !{{[0-9]+}}
pub fn foo19(_: c_uint, _: c_uint) { }
// CHECK: define{{.*}}foo19{{.*}}!type ![[TYPE19:[0-9]+]] !type !{{[0-9]+}}
pub fn foo20(_: c_uint, _: c_uint, _: c_uint) { }
// CHECK: define{{.*}}foo20{{.*}}!type ![[TYPE20:[0-9]+]] !type !{{[0-9]+}}
pub fn foo21(_: c_ushort) { }
// CHECK: define{{.*}}foo21{{.*}}!type ![[TYPE21:[0-9]+]] !type !{{[0-9]+}}
pub fn foo22(_: c_ushort, _: c_ushort) { }
// CHECK: define{{.*}}foo22{{.*}}!type ![[TYPE22:[0-9]+]] !type !{{[0-9]+}}
pub fn foo23(_: c_ushort, _: c_ushort, _: c_ushort) { }
// CHECK: define{{.*}}foo23{{.*}}!type ![[TYPE23:[0-9]+]] !type !{{[0-9]+}}

// CHECK: ![[TYPE0]] = !{i64 0, !"_ZTSFvu2u8E.normalized"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu2u8S_E.normalized"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu2u8S_S_E.normalized"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64}}E.normalized"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64}}{{u3i32|u3i64|S_}}E.normalized"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64}}{{u3i32|u3i64|S_}}{{u3i64|S_|S0_}}E.normalized"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64}}E.normalized"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64}}{{u3u32|u3u64|S_}}E.normalized"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64}}{{u3u32|u3u64|S_}}{{u3u64|S_|S0_}}E.normalized"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu2i8E.normalized"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFv{{u2i8S_|u2u8u2i8}}E.normalized"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFv{{u2i8S_S_|u2u8u2i8S0_}}E.normalized"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64}}E.normalized"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64}}S_E.normalized"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64}}S_S_E.normalized"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvu3i16E.normalized"}
// CHECK: ![[TYPE16]] = !{i64 0, !"_ZTSFvu3i16S_E.normalized"}
// CHECK: ![[TYPE17]] = !{i64 0, !"_ZTSFvu3i16S_S_E.normalized"}
// CHECK: ![[TYPE18]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64}}E.normalized"}
// CHECK: ![[TYPE19]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64}}S_E.normalized"}
// CHECK: ![[TYPE20]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64}}S_S_E.normalized"}
// CHECK: ![[TYPE21]] = !{i64 0, !"_ZTSFvu3u16E.normalized"}
// CHECK: ![[TYPE22]] = !{i64 0, !"_ZTSFvu3u16S_E.normalized"}
// CHECK: ![[TYPE23]] = !{i64 0, !"_ZTSFvu3u16S_S_E.normalized"}
