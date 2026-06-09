// Verifies that type metadata identifiers for functions are emitted correctly
// for primitive types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

extern crate core;
use core::ffi::*;

pub fn foo1(_: ()) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: (), _: c_void) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: (), _: c_void, _: c_void) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: *mut ()) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: *mut (), _: *mut c_void) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: *mut (), _: *mut c_void, _: *mut c_void) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: *const ()) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: *const (), _: *const c_void) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: *const (), _: *const c_void, _: *const c_void) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: bool) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: bool, _: bool) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: bool, _: bool, _: bool) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo13(_: i8) {}
// CHECK: define{{.*}}5foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo14(_: i8, _: i8) {}
// CHECK: define{{.*}}5foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo15(_: i8, _: i8, _: i8) {}
// CHECK: define{{.*}}5foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo16(_: i16) {}
// CHECK: define{{.*}}5foo16{{.*}}!type ![[TYPE16:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo17(_: i16, _: i16) {}
// CHECK: define{{.*}}5foo17{{.*}}!type ![[TYPE17:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo18(_: i16, _: i16, _: i16) {}
// CHECK: define{{.*}}5foo18{{.*}}!type ![[TYPE18:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo19(_: i32) {}
// CHECK: define{{.*}}5foo19{{.*}}!type ![[TYPE19:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo20(_: i32, _: i32) {}
// CHECK: define{{.*}}5foo20{{.*}}!type ![[TYPE20:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo21(_: i32, _: i32, _: i32) {}
// CHECK: define{{.*}}5foo21{{.*}}!type ![[TYPE21:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo22(_: i64) {}
// CHECK: define{{.*}}5foo22{{.*}}!type ![[TYPE22:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo23(_: i64, _: i64) {}
// CHECK: define{{.*}}5foo23{{.*}}!type ![[TYPE23:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo24(_: i64, _: i64, _: i64) {}
// CHECK: define{{.*}}5foo24{{.*}}!type ![[TYPE24:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo25(_: i128) {}
// CHECK: define{{.*}}5foo25{{.*}}!type ![[TYPE25:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo26(_: i128, _: i128) {}
// CHECK: define{{.*}}5foo26{{.*}}!type ![[TYPE26:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo27(_: i128, _: i128, _: i128) {}
// CHECK: define{{.*}}5foo27{{.*}}!type ![[TYPE27:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo28(_: isize) {}
// CHECK: define{{.*}}5foo28{{.*}}!type ![[TYPE28:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo29(_: isize, _: isize) {}
// CHECK: define{{.*}}5foo29{{.*}}!type ![[TYPE29:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo30(_: isize, _: isize, _: isize) {}
// CHECK: define{{.*}}5foo30{{.*}}!type ![[TYPE30:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo31(_: u8) {}
// CHECK: define{{.*}}5foo31{{.*}}!type ![[TYPE31:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo32(_: u8, _: u8) {}
// CHECK: define{{.*}}5foo32{{.*}}!type ![[TYPE32:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo33(_: u8, _: u8, _: u8) {}
// CHECK: define{{.*}}5foo33{{.*}}!type ![[TYPE33:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo34(_: u16) {}
// CHECK: define{{.*}}5foo34{{.*}}!type ![[TYPE34:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo35(_: u16, _: u16) {}
// CHECK: define{{.*}}5foo35{{.*}}!type ![[TYPE35:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo36(_: u16, _: u16, _: u16) {}
// CHECK: define{{.*}}5foo36{{.*}}!type ![[TYPE36:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo37(_: u32) {}
// CHECK: define{{.*}}5foo37{{.*}}!type ![[TYPE37:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo38(_: u32, _: u32) {}
// CHECK: define{{.*}}5foo38{{.*}}!type ![[TYPE38:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo39(_: u32, _: u32, _: u32) {}
// CHECK: define{{.*}}5foo39{{.*}}!type ![[TYPE39:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo40(_: u64) {}
// CHECK: define{{.*}}5foo40{{.*}}!type ![[TYPE40:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo41(_: u64, _: u64) {}
// CHECK: define{{.*}}5foo41{{.*}}!type ![[TYPE41:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo42(_: u64, _: u64, _: u64) {}
// CHECK: define{{.*}}5foo42{{.*}}!type ![[TYPE42:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo43(_: u128) {}
// CHECK: define{{.*}}5foo43{{.*}}!type ![[TYPE43:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo44(_: u128, _: u128) {}
// CHECK: define{{.*}}5foo44{{.*}}!type ![[TYPE44:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo45(_: u128, _: u128, _: u128) {}
// CHECK: define{{.*}}5foo45{{.*}}!type ![[TYPE45:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo46(_: usize) {}
// CHECK: define{{.*}}5foo46{{.*}}!type ![[TYPE46:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo47(_: usize, _: usize) {}
// CHECK: define{{.*}}5foo47{{.*}}!type ![[TYPE47:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo48(_: usize, _: usize, _: usize) {}
// CHECK: define{{.*}}5foo48{{.*}}!type ![[TYPE48:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo49(_: f32) {}
// CHECK: define{{.*}}5foo49{{.*}}!type ![[TYPE49:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo50(_: f32, _: f32) {}
// CHECK: define{{.*}}5foo50{{.*}}!type ![[TYPE50:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo51(_: f32, _: f32, _: f32) {}
// CHECK: define{{.*}}5foo51{{.*}}!type ![[TYPE51:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo52(_: f64) {}
// CHECK: define{{.*}}5foo52{{.*}}!type ![[TYPE52:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo53(_: f64, _: f64) {}
// CHECK: define{{.*}}5foo53{{.*}}!type ![[TYPE53:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo54(_: f64, _: f64, _: f64) {}
// CHECK: define{{.*}}5foo54{{.*}}!type ![[TYPE54:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo55(_: char) {}
// CHECK: define{{.*}}5foo55{{.*}}!type ![[TYPE55:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo56(_: char, _: char) {}
// CHECK: define{{.*}}5foo56{{.*}}!type ![[TYPE56:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo57(_: char, _: char, _: char) {}
// CHECK: define{{.*}}5foo57{{.*}}!type ![[TYPE57:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo58(_: &str) {}
// CHECK: define{{.*}}5foo58{{.*}}!type ![[TYPE58:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo59(_: &str, _: &str) {}
// CHECK: define{{.*}}5foo59{{.*}}!type ![[TYPE59:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo60(_: &str, _: &str, _: &str) {}
// CHECK: define{{.*}}5foo60{{.*}}!type ![[TYPE60:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvvE"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvPvE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvPvS_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvPvS_S_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvPKvE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvPKvS0_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvPKvS0_S0_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvbE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvbbE"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvbbbE"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvu2i8E"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvu2i8S_E"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvu2i8S_S_E"}
// CHECK: ![[TYPE16]] = !{i64 0, !"_ZTSFvu3i16E"}
// CHECK: ![[TYPE17]] = !{i64 0, !"_ZTSFvu3i16S_E"}
// CHECK: ![[TYPE18]] = !{i64 0, !"_ZTSFvu3i16S_S_E"}
// CHECK: ![[TYPE19]] = !{i64 0, !"_ZTSFvu3i32E"}
// CHECK: ![[TYPE20]] = !{i64 0, !"_ZTSFvu3i32S_E"}
// CHECK: ![[TYPE21]] = !{i64 0, !"_ZTSFvu3i32S_S_E"}
// CHECK: ![[TYPE22]] = !{i64 0, !"_ZTSFvu3i64E"}
// CHECK: ![[TYPE23]] = !{i64 0, !"_ZTSFvu3i64S_E"}
// CHECK: ![[TYPE24]] = !{i64 0, !"_ZTSFvu3i64S_S_E"}
// CHECK: ![[TYPE25]] = !{i64 0, !"_ZTSFvu4i128E"}
// CHECK: ![[TYPE26]] = !{i64 0, !"_ZTSFvu4i128S_E"}
// CHECK: ![[TYPE27]] = !{i64 0, !"_ZTSFvu4i128S_S_E"}
// CHECK: ![[TYPE28]] = !{i64 0, !"_ZTSFvu5isizeE"}
// CHECK: ![[TYPE29]] = !{i64 0, !"_ZTSFvu5isizeS_E"}
// CHECK: ![[TYPE30]] = !{i64 0, !"_ZTSFvu5isizeS_S_E"}
// CHECK: ![[TYPE31]] = !{i64 0, !"_ZTSFvu2u8E"}
// CHECK: ![[TYPE32]] = !{i64 0, !"_ZTSFvu2u8S_E"}
// CHECK: ![[TYPE33]] = !{i64 0, !"_ZTSFvu2u8S_S_E"}
// CHECK: ![[TYPE34]] = !{i64 0, !"_ZTSFvu3u16E"}
// CHECK: ![[TYPE35]] = !{i64 0, !"_ZTSFvu3u16S_E"}
// CHECK: ![[TYPE36]] = !{i64 0, !"_ZTSFvu3u16S_S_E"}
// CHECK: ![[TYPE37]] = !{i64 0, !"_ZTSFvu3u32E"}
// CHECK: ![[TYPE38]] = !{i64 0, !"_ZTSFvu3u32S_E"}
// CHECK: ![[TYPE39]] = !{i64 0, !"_ZTSFvu3u32S_S_E"}
// CHECK: ![[TYPE40]] = !{i64 0, !"_ZTSFvu3u64E"}
// CHECK: ![[TYPE41]] = !{i64 0, !"_ZTSFvu3u64S_E"}
// CHECK: ![[TYPE42]] = !{i64 0, !"_ZTSFvu3u64S_S_E"}
// CHECK: ![[TYPE43]] = !{i64 0, !"_ZTSFvu4u128E"}
// CHECK: ![[TYPE44]] = !{i64 0, !"_ZTSFvu4u128S_E"}
// CHECK: ![[TYPE45]] = !{i64 0, !"_ZTSFvu4u128S_S_E"}
// CHECK: ![[TYPE46]] = !{i64 0, !"_ZTSFvu5usizeE"}
// CHECK: ![[TYPE47]] = !{i64 0, !"_ZTSFvu5usizeS_E"}
// CHECK: ![[TYPE48]] = !{i64 0, !"_ZTSFvu5usizeS_S_E"}
// CHECK: ![[TYPE49]] = !{i64 0, !"_ZTSFvfE"}
// CHECK: ![[TYPE50]] = !{i64 0, !"_ZTSFvffE"}
// CHECK: ![[TYPE51]] = !{i64 0, !"_ZTSFvfffE"}
// CHECK: ![[TYPE52]] = !{i64 0, !"_ZTSFvdE"}
// CHECK: ![[TYPE53]] = !{i64 0, !"_ZTSFvddE"}
// CHECK: ![[TYPE54]] = !{i64 0, !"_ZTSFvdddE"}
// CHECK: ![[TYPE55]] = !{i64 0, !"_ZTSFvu4charE"}
// CHECK: ![[TYPE56]] = !{i64 0, !"_ZTSFvu4charS_E"}
// CHECK: ![[TYPE57]] = !{i64 0, !"_ZTSFvu4charS_S_E"}
// CHECK: ![[TYPE58]] = !{i64 0, !"_ZTSFvu3refIu3strEE"}
// CHECK: ![[TYPE59]] = !{i64 0, !"_ZTSFvu3refIu3strES0_E"}
// CHECK: ![[TYPE60]] = !{i64 0, !"_ZTSFvu3refIu3strES0_S0_E"}
