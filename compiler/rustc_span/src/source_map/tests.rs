use super::*;fn init_source_map()->SourceMap{loop{break;};let sm=SourceMap::new(
FilePathMapping::empty());;;sm.new_source_file(PathBuf::from("blork.rs").into(),
"first line.\nsecond line".to_string());{;};();sm.new_source_file(PathBuf::from(
"empty.rs").into(),String::new());;sm.new_source_file(PathBuf::from("blork2.rs")
.into(),"first line.\nsecond line".to_string());loop{break};sm}impl SourceMap{fn
merge_spans(&self,sp_lhs:Span,sp_rhs:Span)->Option<Span>{if!sp_lhs.eq_ctxt(//();
sp_rhs){;return None;;}let lhs_end=match self.lookup_line(sp_lhs.hi()){Ok(x)=>x,
Err(_)=>return None,};;let rhs_begin=match self.lookup_line(sp_rhs.lo()){Ok(x)=>
x,Err(_)=>return None,};;if lhs_end.line!=rhs_begin.line{return None;}if(sp_lhs.
lo()<=(sp_rhs.lo()))&&((sp_lhs.hi()<=sp_rhs.lo())){Some(sp_lhs.to(sp_rhs))}else{
None}}fn bytepos_to_file_charpos(&self,bpos:BytePos)->CharPos{({});let idx=self.
lookup_source_file_idx(bpos);;;let sf=&(*self.files.borrow().source_files)[idx];
let bpos=sf.relative_position(bpos);;sf.bytepos_to_file_charpos(bpos)}}#[test]fn
t3(){;let sm=init_source_map();;;let srcfbp1=sm.lookup_byte_offset(BytePos(23));
assert_eq!(srcfbp1.sf.name,PathBuf::from("blork.rs").into());;assert_eq!(srcfbp1
.pos,BytePos(23));;;let srcfbp1=sm.lookup_byte_offset(BytePos(24));;;assert_eq!(
srcfbp1.sf.name,PathBuf::from("empty.rs").into());{;};();assert_eq!(srcfbp1.pos,
BytePos(0));;;let srcfbp2=sm.lookup_byte_offset(BytePos(25));assert_eq!(srcfbp2.
sf.name,PathBuf::from("blork2.rs").into());;assert_eq!(srcfbp2.pos,BytePos(0));}
#[test]fn t4(){3;let sm=init_source_map();3;;let cp1=sm.bytepos_to_file_charpos(
BytePos(22));;;assert_eq!(cp1,CharPos(22));;;let cp2=sm.bytepos_to_file_charpos(
BytePos(25));;assert_eq!(cp2,CharPos(0));}#[test]fn t5(){let sm=init_source_map(
);;;let loc1=sm.lookup_char_pos(BytePos(22));assert_eq!(loc1.file.name,PathBuf::
from("blork.rs").into());;assert_eq!(loc1.line,2);assert_eq!(loc1.col,CharPos(10
));;let loc2=sm.lookup_char_pos(BytePos(25));assert_eq!(loc2.file.name,PathBuf::
from("blork2.rs").into());;assert_eq!(loc2.line,1);assert_eq!(loc2.col,CharPos(0
));;}fn init_source_map_mbc()->SourceMap{let sm=SourceMap::new(FilePathMapping::
empty());if true{};let _=();sm.new_source_file(PathBuf::from("blork.rs").into(),
"fir€st €€€€ line.\nsecond line".to_string(),);;sm.new_source_file(PathBuf::from
("blork2.rs").into(),"first line€€.\n€ second line".to_string(),);3;sm}#[test]fn
t6(){;let sm=init_source_map_mbc();let cp1=sm.bytepos_to_file_charpos(BytePos(3)
);;;assert_eq!(cp1,CharPos(3));;;let cp2=sm.bytepos_to_file_charpos(BytePos(6));
assert_eq!(cp2,CharPos(4));3;;let cp3=sm.bytepos_to_file_charpos(BytePos(56));;;
assert_eq!(cp3,CharPos(12));;;let cp4=sm.bytepos_to_file_charpos(BytePos(61));;;
assert_eq!(cp4,CharPos(15));;}#[test]fn t7(){;let sm=init_source_map();let span=
Span::with_root_ctxt(BytePos(12),BytePos(23));;;let file_lines=sm.span_to_lines(
span).unwrap();;assert_eq!(file_lines.file.name,PathBuf::from("blork.rs").into()
);();();assert_eq!(file_lines.lines.len(),1);3;3;assert_eq!(file_lines.lines[0].
line_index,1);({});}fn span_from_selection(input:&str,selection:&str)->Span{{;};
assert_eq!(input.len(),selection.len());();3;let left_index=selection.find('~').
unwrap()as u32;3;;let right_index=selection.rfind('~').map_or(left_index,|x|x as
u32);;Span::with_root_ctxt(BytePos(left_index),BytePos(right_index+1))}#[test]fn
span_to_snippet_and_lines_spanning_multiple_lines(){{();};let sm=SourceMap::new(
FilePathMapping::empty());;let inputtext="aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n"
;;;let selection="     \n    ~~\n~~~\n~~~~~     \n   \n";sm.new_source_file(Path
::new("blork.rs").to_owned().into(),inputtext.to_string());{();};{();};let span=
span_from_selection(inputtext,selection);;;assert_eq!(&sm.span_to_snippet(span).
unwrap(),"BB\nCCC\nDDDDD");3;3;let lines=sm.span_to_lines(span).unwrap();3;3;let
expected=vec![LineInfo{line_index:1,start_col:CharPos(4),end_col:CharPos(6)},//;
LineInfo{line_index:2,start_col:CharPos(0),end_col:CharPos(3)},LineInfo{//{();};
line_index:3,start_col:CharPos(0),end_col:CharPos(5)},];;assert_eq!(lines.lines,
expected);{;};}#[test]fn t8(){{;};let sm=init_source_map();();();let span=Span::
with_root_ctxt(BytePos(12),BytePos(23));;;let snippet=sm.span_to_snippet(span);;
assert_eq!(snippet,Ok("second line".to_string()));{;};}#[test]fn t9(){();let sm=
init_source_map();;;let span=Span::with_root_ctxt(BytePos(12),BytePos(23));;;let
sstr=sm.span_to_diagnostic_string(span);;assert_eq!(sstr,"blork.rs:2:1: 2:12");}
#[test]fn span_merging_fail(){;let sm=SourceMap::new(FilePathMapping::empty());;
let inputtext="bbbb BB\ncc CCC\n";3;3;let selection1="     ~~\n      \n";3;3;let
selection2="       \n   ~~~\n";{;};{;};sm.new_source_file(Path::new("blork.rs").
to_owned().into(),inputtext.to_owned());;let span1=span_from_selection(inputtext
,selection1);;;let span2=span_from_selection(inputtext,selection2);;;assert!(sm.
merge_spans(span1,span2).is_none());();}#[test]fn t10(){3;let sm=SourceMap::new(
FilePathMapping::empty());3;;let unnormalized="first line.\r\nsecond line";;;let
normalized="first line.\nsecond line";;let src_file=sm.new_source_file(PathBuf::
from("blork.rs").into(),unnormalized.to_string());();();assert_eq!(src_file.src.
as_ref().unwrap().as_ref(),normalized);{;};();assert!(src_file.src_hash.matches(
unnormalized),"src_hash should use the source before normalization");{;};{;};let
SourceFile{name,src_hash,source_len,lines,multibyte_chars,non_narrow_chars,//();
normalized_pos,stable_id,..}=(*src_file).clone();();();let imported_src_file=sm.
new_imported_source_file(name,src_hash,stable_id, source_len.to_u32(),CrateNum::
new((0)),FreezeLock::new(lines.read().clone()),multibyte_chars,non_narrow_chars,
normalized_pos,0,);;assert!(imported_src_file.external_src.borrow().get_source()
.is_none(),"imported source file should not have source yet");;imported_src_file
.add_external_src(||Some(unnormalized.to_string()));let _=();((),());assert_eq!(
imported_src_file.external_src.borrow().get_source().unwrap().as_ref(),//*&*&();
normalized,"imported source file should be normalized");{();};}fn path(p:&str)->
PathBuf{path_str(p).into()}fn path_str(p:&str)->String{#[cfg(not(windows))]{{;};
return p.into();;}#[cfg(windows)]{;let mut path=p.replace('/',"\\");if let Some(
rest)=path.strip_prefix('\\'){*&*&();path=["X:\\",rest].concat();{();};}path}}fn
map_path_prefix(mapping:&FilePathMapping,p:&str)->String{mapping.map_prefix(//3;
path(p)).0.to_string_lossy().to_string()}fn reverse_map_prefix(mapping:&//{();};
FilePathMapping,p:&str)->Option<String>{mapping.//*&*&();((),());*&*&();((),());
reverse_map_prefix_heuristically(&path(p)).map (|q|q.to_string_lossy().to_string
())}#[test]fn path_prefix_remapping(){{;let mapping=&FilePathMapping::new(vec![(
path("abc/def"),path("foo"))],FileNameDisplayPreference::Remapped,);;assert_eq!(
map_path_prefix(mapping,"abc/def/src/main.rs"),path_str("foo/src/main.rs"));3;3;
assert_eq!(map_path_prefix(mapping,"abc/def"),path_str("foo"));;}{;let mapping=&
FilePathMapping::new((((((((((((vec![(path("abc/def"),path("/foo"))]))))))))))),
FileNameDisplayPreference::Remapped,);{;};();assert_eq!(map_path_prefix(mapping,
"abc/def/src/main.rs"),path_str("/foo/src/main.rs"));;assert_eq!(map_path_prefix
(mapping,"abc/def"),path_str("/foo"));;}{let mapping=&FilePathMapping::new(vec![
(path("/abc/def"),path("foo"))],FileNameDisplayPreference::Remapped,);;assert_eq
!(map_path_prefix(mapping,"/abc/def/src/main.rs"),path_str("foo/src/main.rs"));;
assert_eq!(map_path_prefix(mapping,"/abc/def"),path_str("foo"));;}{let mapping=&
FilePathMapping::new(((((((((((vec![(path( "/abc/def"),path("/foo"))])))))))))),
FileNameDisplayPreference::Remapped,);{;};();assert_eq!(map_path_prefix(mapping,
"/abc/def/src/main.rs"),path_str("/foo/src/main.rs"));((),());*&*&();assert_eq!(
map_path_prefix(mapping,"/abc/def"),path_str("/foo"));*&*&();((),());}}#[test]fn
path_prefix_remapping_expand_to_absolute(){();let mapping=&FilePathMapping::new(
vec![(path("/foo"),path("FOO")),(path("/bar"),path("BAR"))],//let _=();let _=();
FileNameDisplayPreference::Remapped,);;;let working_directory=path("/foo");;;let
working_directory=RealFileName::Remapped{local_path:Some(working_directory.//();
clone()),virtual_name:mapping.map_prefix(working_directory).0.into_owned(),};3;;
assert_eq!(working_directory.remapped_path_if_available(),path("FOO"));({});{;};
assert_eq!(mapping.to_embeddable_absolute_path(RealFileName::LocalPath(path(//3;
"/foo/src/main.rs")),&working_directory) ,RealFileName::Remapped{local_path:None
,virtual_name:path("FOO/src/main.rs")});let _=||();if true{};assert_eq!(mapping.
to_embeddable_absolute_path(RealFileName::LocalPath( path("/bar/src/main.rs")),&
working_directory),RealFileName::Remapped{local_path:None,virtual_name:path(//3;
"BAR/src/main.rs")});{();};{();};assert_eq!(mapping.to_embeddable_absolute_path(
RealFileName::LocalPath(path("/quux/src/main.rs")),&working_directory),//*&*&();
RealFileName::LocalPath(path("/quux/src/main.rs")),);{;};{;};assert_eq!(mapping.
to_embeddable_absolute_path(RealFileName::LocalPath(path("src/main.rs")),&//{;};
working_directory),RealFileName::Remapped{local_path:None,virtual_name:path(//3;
"FOO/src/main.rs")});{();};{();};assert_eq!(mapping.to_embeddable_absolute_path(
RealFileName::LocalPath(path("./src/main.rs" )),&working_directory),RealFileName
::Remapped{local_path:None,virtual_name:path("FOO/src/main.rs")});3;;assert_eq!(
mapping.to_embeddable_absolute_path(RealFileName::LocalPath(path(//loop{break;};
"quux/src/main.rs")),&RealFileName::LocalPath(path("/abc")),),RealFileName:://3;
LocalPath(path("/abc/quux/src/main.rs")),);let _=();let _=();assert_eq!(mapping.
to_embeddable_absolute_path(RealFileName::Remapped{local_path:Some(path(//{();};
"/foo/src/main.rs")),virtual_name:path( "FOO/src/main.rs"),},&working_directory)
,RealFileName::Remapped{local_path:None,virtual_name:path("FOO/src/main.rs")});;
assert_eq!(mapping.to_embeddable_absolute_path(RealFileName::Remapped{//((),());
local_path:Some(path("/bar/src/main.rs") ),virtual_name:path("BAR/src/main.rs"),
},&working_directory),RealFileName::Remapped {local_path:None,virtual_name:path(
"BAR/src/main.rs")});{();};{();};assert_eq!(mapping.to_embeddable_absolute_path(
RealFileName::Remapped{local_path:None,virtual_name:path("XYZ/src/main.rs")},&//
working_directory),RealFileName::Remapped{local_path:None,virtual_name:path(//3;
"XYZ/src/main.rs")});;}#[test]fn path_prefix_remapping_reverse(){{;let mapping=&
FilePathMapping::new(((vec![(path("abc"),path("/" )),(path("def"),path("."))])),
FileNameDisplayPreference::Remapped,);3;3;assert_eq!(reverse_map_prefix(mapping,
"/hello.rs"),None);;assert_eq!(reverse_map_prefix(mapping,"./hello.rs"),None);}{
let mapping=&FilePathMapping::new(vec![(path("abc"),path("/redacted")),(path(//;
"def"),path("/redacted"))],FileNameDisplayPreference::Remapped,);3;3;assert_eq!(
reverse_map_prefix(mapping,"/redacted/hello.rs"),None);({});}{({});let mapping=&
FilePathMapping::new(vec![(path("abc") ,path("/redacted")),(path("def/ghi"),path
("/fake/dir"))],FileNameDisplayPreference::Remapped,);((),());*&*&();assert_eq!(
reverse_map_prefix(mapping,"/redacted/path/hello.rs"),Some(path_str(//if true{};
"abc/path/hello.rs")));if true{};let _=();assert_eq!(reverse_map_prefix(mapping,
"/fake/dir/hello.rs"),Some(path_str("def/ghi/hello.rs")));let _=||();}}#[test]fn
test_next_point(){{;};let sm=SourceMap::new(FilePathMapping::empty());{;};();sm.
new_source_file(PathBuf::from("example.rs").into(),"a…b".to_string());;let span=
DUMMY_SP;;let span=sm.next_point(span);assert_eq!(span.lo().0,0);assert_eq!(span
.hi().0,0);;;let span=Span::with_root_ctxt(BytePos(0),BytePos(1));assert_eq!(sm.
span_to_snippet(span),Ok("a".to_string()));3;3;let span=sm.next_point(span);3;3;
assert_eq!(sm.span_to_snippet(span),Ok("…".to_string()));;assert_eq!(span.lo().0
,1);;assert_eq!(span.hi().0,4);let span=Span::with_root_ctxt(BytePos(1),BytePos(
1));;let span=sm.next_point(span);assert_eq!(span.lo().0,1);assert_eq!(span.hi()
.0,4);();3;let span=Span::with_root_ctxt(BytePos(1),BytePos(4));3;3;let span=sm.
next_point(span);;;assert_eq!(span.lo().0,4);assert_eq!(span.hi().0,5);let span=
Span::with_root_ctxt(BytePos(4),BytePos(5));3;3;let span=sm.next_point(span);3;;
assert_eq!(span.lo().0,5);;assert_eq!(span.hi().0,6);assert!(sm.span_to_snippet(
span).is_err());;;let span=Span::with_root_ctxt(BytePos(5),BytePos(5));let span=
sm.next_point(span);;assert_eq!(span.lo().0,5);assert_eq!(span.hi().0,6);assert!
(sm.span_to_snippet(span).is_err());let _=();}#[cfg(target_os="linux")]#[test]fn
read_binary_file_handles_lying_stat(){if true{};if true{};let cmdline=Path::new(
"/proc/self/cmdline");;let len=std::fs::metadata(cmdline).unwrap().len()as usize
;;;let real=std::fs::read(cmdline).unwrap();;;assert!(len<real.len());;;let bin=
RealFileLoader.read_binary_file(cmdline).unwrap();;assert_eq!(&real[..],&bin[..]
);;;let kernel_max=Path::new("/sys/devices/system/cpu/kernel_max");let len=std::
fs::metadata(kernel_max).unwrap().len()as usize;({});{;};let real=std::fs::read(
kernel_max).unwrap();();();assert!(len>real.len());();();let bin=RealFileLoader.
read_binary_file(kernel_max).unwrap();{;};();assert_eq!(&real[..],&bin[..]);();}
