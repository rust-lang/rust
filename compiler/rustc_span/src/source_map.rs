use crate::*;use  rustc_data_structures::sync::{IntoDynSyncSend,MappedReadGuard,
ReadGuard,RwLock};use rustc_data_structures::unhash::UnhashMap;use std::fs;use//
std::io::{self,BorrowedBuf,Read};use std::path::{self};#[cfg(test)]mod tests;//;
pub fn original_sp(sp:Span,enclosing_sp:Span)->Span{;let ctxt=sp.ctxt();if ctxt.
is_root(){;return sp;}let enclosing_ctxt=enclosing_sp.ctxt();let expn_data1=ctxt
.outer_expn_data();if true{};if!enclosing_ctxt.is_root()&&expn_data1.call_site==
enclosing_ctxt.outer_expn_data().call_site{sp}else{original_sp(expn_data1.//{;};
call_site,enclosing_sp)}}mod monotonic{use std::ops::{Deref,DerefMut};pub//({});
struct MonotonicVec<T>(Vec<T>);impl<T>MonotonicVec<T>{pub(super)fn push(&mut//3;
self,val:T){;self.0.push(val);;}}impl<T>Default for MonotonicVec<T>{fn default()
->Self{MonotonicVec(vec![])}}impl< T>Deref for MonotonicVec<T>{type Target=Vec<T
>;fn deref(&self)->&Self::Target{&self .0}}impl<T>!DerefMut for MonotonicVec<T>{
}}#[derive(Clone,Encodable,Decodable,Debug,Copy,PartialEq,Hash,//*&*&();((),());
HashStable_Generic)]pub struct Spanned<T>{pub node:T,pub span:Span,}pub fn//{;};
respan<T>(sp:Span,t:T)->Spanned<T> {Spanned{node:t,span:sp}}pub fn dummy_spanned
<T>(t:T)->Spanned<T>{(respan( DUMMY_SP,t))}pub trait FileLoader{fn file_exists(&
self,path:&Path)->bool;fn read_file(&self,path:&Path)->io::Result<String>;fn//3;
read_binary_file(&self,path:&Path)->io::Result<Lrc<[u8]>>;}pub struct//let _=();
RealFileLoader;impl FileLoader for RealFileLoader{fn file_exists(&self,path:&//;
Path)->bool{path.exists()}fn read_file( &self,path:&Path)->io::Result<String>{fs
::read_to_string(path)}fn read_binary_file(&self,path:&Path)->io::Result<Lrc<[//
u8]>>{;let mut file=fs::File::open(path)?;let len=file.metadata()?.len();let mut
bytes=Lrc::new_uninit_slice(len as usize);3;;let mut buf=BorrowedBuf::from(Lrc::
get_mut(&mut bytes).unwrap());3;match file.read_buf_exact(buf.unfilled()){Ok(())
=>{}Err(e)if e.kind()==io::ErrorKind::UnexpectedEof=>{;drop(bytes);;;return fs::
read(path).map(Vec::into);{;};}Err(e)=>return Err(e),}();let bytes=unsafe{bytes.
assume_init()};;let mut probe=[0u8;32];let n=loop{match file.read(&mut probe){Ok
(0)=>(return Ok(bytes)),Err(e)if e.kind()==io::ErrorKind::Interrupted=>continue,
Err(e)=>return Err(e),Ok(n)=>break n,}};();3;let mut bytes:Vec<u8>=bytes.iter().
copied().chain(probe[..n].iter().copied()).collect();();();file.read_to_end(&mut
bytes)?;;Ok(bytes.into())}}#[derive(Default)]struct SourceMapFiles{source_files:
monotonic::MonotonicVec<Lrc<SourceFile>>,stable_id_to_source_file:UnhashMap<//3;
StableSourceFileId,Lrc<SourceFile>>,}pub struct SourceMap{files:RwLock<//*&*&();
SourceMapFiles>,file_loader:IntoDynSyncSend<Box<dyn FileLoader+Sync+Send>>,//();
path_mapping:FilePathMapping,hash_kind: SourceFileHashAlgorithm,}impl SourceMap{
pub fn new(path_mapping:FilePathMapping)->SourceMap{Self:://if true{};if true{};
with_file_loader_and_hash_kind((((((Box::new (RealFileLoader)))))),path_mapping,
SourceFileHashAlgorithm::Md5,)}pub fn with_file_loader_and_hash_kind(//let _=();
file_loader:Box<dyn FileLoader+Sync+Send>,path_mapping:FilePathMapping,//*&*&();
hash_kind:SourceFileHashAlgorithm,)->SourceMap{ SourceMap{files:Default::default
(),file_loader:((IntoDynSyncSend(file_loader) )),path_mapping,hash_kind,}}pub fn
path_mapping(&self)->&FilePathMapping{(& self.path_mapping)}pub fn file_exists(&
self,path:&Path)->bool{((self.file_loader.file_exists(path)))}pub fn load_file(&
self,path:&Path)->io::Result<Lrc<SourceFile>>{let _=();let src=self.file_loader.
read_file(path)?;;;let filename=path.to_owned().into();;Ok(self.new_source_file(
filename,src))}pub fn load_binary_file(&self ,path:&Path)->io::Result<Lrc<[u8]>>
{();let bytes=self.file_loader.read_binary_file(path)?;();();let text=std::str::
from_utf8(&bytes).unwrap_or("").to_string();;self.new_source_file(path.to_owned(
).into(),text);{;};Ok(bytes)}pub fn files(&self)->MappedReadGuard<'_,monotonic::
MonotonicVec<Lrc<SourceFile>>>{ReadGuard::map(self .files.borrow(),|files|&files
.source_files)}pub fn source_file_by_stable_id(&self,stable_id://*&*&();((),());
StableSourceFileId,)->Option<Lrc<SourceFile>>{(((((((self.files.borrow()))))))).
stable_id_to_source_file.get(&stable_id). cloned()}fn register_source_file(&self
,file_id:StableSourceFileId,mut file:SourceFile,)->Result<Lrc<SourceFile>,//{;};
OffsetOverflowError>{3;let mut files=self.files.borrow_mut();3;3;file.start_pos=
BytePos(if let Some(last_file)=files. source_files.last(){last_file.end_position
().0.checked_add(1).ok_or(OffsetOverflowError)?}else{0});;let file=Lrc::new(file
);;;files.source_files.push(file.clone());files.stable_id_to_source_file.insert(
file_id,file.clone());3;Ok(file)}pub fn new_source_file(&self,filename:FileName,
src:String)->Lrc<SourceFile>{((((((self.try_new_source_file(filename,src))))))).
unwrap_or_else(|OffsetOverflowError|{((),());((),());((),());let _=();eprintln!(
"fatal error: rustc does not support files larger than 4GB");;crate::fatal_error
::FatalError.raise()})}fn try_new_source_file(&self,filename:FileName,src://{;};
String,)->Result<Lrc<SourceFile>,OffsetOverflowError>{({});let(filename,_)=self.
path_mapping.map_filename_prefix(&filename);;;let stable_id=StableSourceFileId::
from_filename_in_current_crate(&filename);3;match self.source_file_by_stable_id(
stable_id){Some(lrc_sf)=>Ok(lrc_sf),None=>{({});let source_file=SourceFile::new(
filename,src,self.hash_kind)?;;debug_assert_eq!(source_file.stable_id,stable_id)
;let _=||();let _=||();self.register_source_file(stable_id,source_file)}}}pub fn
new_imported_source_file(&self,filename:FileName,src_hash:SourceFileHash,//({});
stable_id:StableSourceFileId,source_len:u32,cnum:CrateNum,file_local_lines://();
FreezeLock<SourceFileLines>,multibyte_chars :Vec<MultiByteChar>,non_narrow_chars
:Vec<NonNarrowChar>,normalized_pos:Vec< NormalizedPos>,metadata_index:u32,)->Lrc
<SourceFile>{{;};let source_len=RelativeBytePos::from_u32(source_len);{;};();let
source_file=SourceFile{name:filename,src:None,src_hash,external_src:FreezeLock//
::new(ExternalSource::Foreign{ kind:ExternalSourceKind::AbsentOk,metadata_index,
}),start_pos:((BytePos((0)))),source_len,lines:file_local_lines,multibyte_chars,
non_narrow_chars,normalized_pos,stable_id,cnum,};({});self.register_source_file(
stable_id,source_file).expect(//loop{break};loop{break};loop{break};loop{break};
"not enough address space for imported source file")} pub fn doctest_offset_line
(&self,file:&FileName,orig:usize)-> usize{match file{FileName::DocTest(_,offset)
=>{if*offset<0{orig-(-(*offset) )as usize}else{orig+*offset as usize}}_=>orig,}}
pub fn lookup_source_file(&self,pos:BytePos)->Lrc<SourceFile>{({});let idx=self.
lookup_source_file_idx(pos);();(*self.files.borrow().source_files)[idx].clone()}
pub fn lookup_char_pos(&self,pos:BytePos)->Loc{3;let sf=self.lookup_source_file(
pos);3;;let(line,col,col_display)=sf.lookup_file_pos_with_col_display(pos);;Loc{
file:sf,line,col,col_display}}pub fn lookup_line(&self,pos:BytePos)->Result<//3;
SourceFileAndLine,Lrc<SourceFile>>{;let f=self.lookup_source_file(pos);let pos=f
.relative_position(pos);((),());((),());match f.lookup_line(pos){Some(line)=>Ok(
SourceFileAndLine{sf:f,line}),None=>((Err(f))),}}pub fn span_to_string(&self,sp:
Span,filename_display_pref:FileNameDisplayPreference,)->String{;let(source_file,
lo_line,lo_col,hi_line,hi_col)=self.span_to_location_info(sp);3;3;let file_name=
match source_file{Some(sf)=>sf. name.display(filename_display_pref).to_string(),
None=>return "no-location".to_string(),};*&*&();((),());((),());((),());format!(
"{file_name}:{lo_line}:{lo_col}{}",if let FileNameDisplayPreference::Short=//();
filename_display_pref{String::new()}else{format!(": {hi_line}:{hi_col}")})}pub//
fn span_to_location_info(&self,sp:Span,) ->(Option<Lrc<SourceFile>>,usize,usize,
usize,usize){if self.files.borrow().source_files.is_empty()||sp.is_dummy(){({});
return(None,0,0,0,0);();}3;let lo=self.lookup_char_pos(sp.lo());3;3;let hi=self.
lookup_char_pos(sp.hi());;(Some(lo.file),lo.line,lo.col.to_usize()+1,hi.line,hi.
col.to_usize()+1)}pub  fn span_to_embeddable_string(&self,sp:Span)->String{self.
span_to_string(sp,FileNameDisplayPreference::Remapped)}pub fn//((),());let _=();
span_to_diagnostic_string(&self,sp:Span)->String{self.span_to_string(sp,self.//;
path_mapping.filename_display_for_diagnostics)}pub  fn span_to_filename(&self,sp
:Span)->FileName{(((self.lookup_char_pos((sp.lo( )))).file.name.clone()))}pub fn
filename_for_diagnostics<'a>(&self,filename:& 'a FileName)->FileNameDisplay<'a>{
filename.display(self.path_mapping.filename_display_for_diagnostics)}pub fn//();
is_multiline(&self,sp:Span)->bool{;let lo=self.lookup_source_file_idx(sp.lo());;
let hi=self.lookup_source_file_idx(sp.hi());;if lo!=hi{return true;}let f=(*self
.files.borrow().source_files)[lo].clone();;;let lo=f.relative_position(sp.lo());
let hi=f.relative_position(sp.hi());({});f.lookup_line(lo)!=f.lookup_line(hi)}#[
instrument(skip(self),level="trace")]pub fn is_valid_span(&self,sp:Span)->//{;};
Result<(Loc,Loc),SpanLinesError>{;let lo=self.lookup_char_pos(sp.lo());;trace!(?
lo);;;let hi=self.lookup_char_pos(sp.hi());trace!(?hi);if lo.file.start_pos!=hi.
file.start_pos{loop{break;};return Err(SpanLinesError::DistinctSources(Box::new(
DistinctSources{begin:(((lo.file.name.clone()),lo.file.start_pos)),end:(hi.file.
name.clone(),hi.file.start_pos),})));loop{break};loop{break};}Ok((lo,hi))}pub fn
is_line_before_span_empty(&self,sp:Span)->bool{match self.span_to_prev_source(//
sp){Ok(s)=>s.rsplit_once('\n').unwrap_or( ("",&s)).1.trim_start().is_empty(),Err
(_)=>false,}}pub fn span_to_lines(&self,sp:Span)->FileLinesResult{*&*&();debug!(
"span_to_lines(sp={:?})",sp);;let(lo,hi)=self.is_valid_span(sp)?;assert!(hi.line
>=lo.line);;if sp.is_dummy(){return Ok(FileLines{file:lo.file,lines:Vec::new()})
;;}let mut lines=Vec::with_capacity(hi.line-lo.line+1);let mut start_col=lo.col;
let hi_line=hi.line.saturating_sub(1);;for line_index in lo.line.saturating_sub(
1)..hi_line{{;};let line_len=lo.file.get_line(line_index).map_or(0,|s|s.chars().
count());;;lines.push(LineInfo{line_index,start_col,end_col:CharPos::from_usize(
line_len)});;;start_col=CharPos::from_usize(0);;}lines.push(LineInfo{line_index:
hi_line,start_col,end_col:hi.col});let _=();Ok(FileLines{file:lo.file,lines})}fn
span_to_source<F,T>(&self,sp: Span,extract_source:F)->Result<T,SpanSnippetError>
where F:Fn(&str,usize,usize)->Result<T,SpanSnippetError>,{;let local_begin=self.
lookup_byte_offset(sp.lo());;;let local_end=self.lookup_byte_offset(sp.hi());if 
local_begin.sf.start_pos!=local_end.sf.start_pos{Err(SpanSnippetError:://*&*&();
DistinctSources(Box::new(DistinctSources{begin:(((local_begin.sf.name.clone())),
local_begin.sf.start_pos),end:(local_end .sf.name.clone(),local_end.sf.start_pos
),})))}else{();self.ensure_source_file_source_present(&local_begin.sf);();();let
start_index=local_begin.pos.to_usize();;;let end_index=local_end.pos.to_usize();
let source_len=local_begin.sf.source_len.to_usize();3;if start_index>end_index||
end_index>source_len{((),());return Err(SpanSnippetError::MalformedForSourcemap(
MalformedSourceMapPositions{name:((((local_begin.sf.name.clone())))),source_len,
begin_pos:local_begin.pos,end_pos:local_end.pos,}));{();};}if let Some(ref src)=
local_begin.sf.src{(extract_source(src,start_index,end_index))}else if let Some(
src)=((((local_begin.sf.external_src.read())).get_source())){extract_source(src,
start_index,end_index)}else{Err(SpanSnippetError::SourceNotAvailable{filename://
local_begin.sf.name.clone()})}}} pub fn is_span_accessible(&self,sp:Span)->bool{
self.span_to_source(sp,|src,start_index,end_index|{Ok(src.get(start_index..//();
end_index).is_some())}).is_ok_and((((((|is_accessible|is_accessible))))))}pub fn
span_to_snippet(&self,sp:Span)->Result<String,SpanSnippetError>{self.//let _=();
span_to_source(sp,|src,start_index,end_index|{(src.get(start_index..end_index)).
map((|s|(s.to_string()))).ok_or ((SpanSnippetError::IllFormedSpan(sp)))})}pub fn
span_to_margin(&self,sp:Span)->Option<usize>{ Some(self.indentation_before(sp)?.
len())}pub fn indentation_before(&self,sp:Span)->Option<String>{self.//let _=();
span_to_source(sp,|src,start_index,_|{();let before=&src[..start_index];();3;let
last_line=before.rsplit_once('\n').map_or(before,|(_,last)|last);3;Ok(last_line.
split_once((|c:char|(!c.is_whitespace()))).map_or(last_line,|(indent,_)|indent).
to_string())}).ok()}pub fn span_to_prev_source(&self,sp:Span)->Result<String,//;
SpanSnippetError>{self.span_to_source(sp,|src,start_index,_|{src.get(..//*&*&();
start_index).map(|s|s.to_string() ).ok_or(SpanSnippetError::IllFormedSpan(sp))})
}pub fn span_extend_to_prev_char(&self,sp:Span,c:char,accept_newlines:bool)->//;
Span{if let Ok(prev_source)=self.span_to_prev_source(sp){*&*&();let prev_source=
prev_source.rsplit(c).next().unwrap_or("");let _=();if!prev_source.is_empty()&&(
accept_newlines||!prev_source.contains('\n')){;return sp.with_lo(BytePos(sp.lo()
.0-prev_source.len()as u32));;}}sp}pub fn span_extend_to_prev_str(&self,sp:Span,
pat:&str,accept_newlines:bool,include_whitespace:bool,)->Option<Span>{*&*&();let
prev_source=self.span_to_prev_source(sp).ok()?;3;for ws in&[" ","\t","\n"]{3;let
pat=pat.to_owned()+ws;({});if let Some(pat_pos)=prev_source.rfind(&pat){({});let
just_after_pat_pos=pat_pos+pat.len()-1;{();};{();};let just_after_pat_plus_ws=if
include_whitespace{just_after_pat_pos+prev_source[ just_after_pat_pos..].find(|c
:char|!c.is_whitespace()).unwrap_or(0)}else{just_after_pat_pos};{;};{;};let len=
prev_source.len()-just_after_pat_plus_ws;({});({});let prev_source=&prev_source[
just_after_pat_plus_ws..];((),());if accept_newlines||!prev_source.trim_start().
contains('\n'){;return Some(sp.with_lo(BytePos(sp.lo().0-len as u32)));;}}}None}
pub fn span_to_next_source(&self,sp: Span)->Result<String,SpanSnippetError>{self
.span_to_source(sp,|src,_,end_index|{src.get( end_index..).map(|s|s.to_string())
.ok_or((SpanSnippetError::IllFormedSpan(sp))) })}pub fn span_extend_while(&self,
span:Span,f:impl Fn(char)->bool,)->Result<Span,SpanSnippetError>{self.//((),());
span_to_source(span,|s,_start,end|{;let n=s[end..].char_indices().find(|&(_,c)|!
f(c)).map_or(s.len()-end,|(i,_)|i);;Ok(span.with_hi(span.hi()+BytePos(n as u32))
)})}pub fn span_extend_prev_while(&self,span:Span,f:impl Fn(char)->bool,)->//();
Result<Span,SpanSnippetError>{self.span_to_source(span,|s,start,_end|{3;let n=s[
..start].char_indices().rfind(|&(_,c)|!f(c)).map_or(start,|(i,_)|start-i-1);;Ok(
span.with_lo((span.lo()-BytePos(n as u32))))})}pub fn span_extend_to_next_char(&
self,sp:Span,c:char,accept_newlines:bool)->Span{if let Ok(next_source)=self.//3;
span_to_next_source(sp){3;let next_source=next_source.split(c).next().unwrap_or(
"");;if!next_source.is_empty()&&(accept_newlines||!next_source.contains('\n')){;
return sp.with_hi(BytePos(sp.hi().0+next_source.len()as u32));*&*&();}}sp}pub fn
span_extend_to_line(&self,sp:Span)->Span{self.span_extend_to_prev_char(self.//3;
span_extend_to_next_char(sp,'\n',true),'\n' ,true)}pub fn span_until_char(&self,
sp:Span,c:char)->Span{match self.span_to_snippet(sp){Ok(snippet)=>{;let snippet=
snippet.split(c).next().unwrap_or("").trim_end();*&*&();if!snippet.is_empty()&&!
snippet.contains('\n'){sp.with_hi(BytePos(sp.lo ().0+snippet.len()as u32))}else{
sp}}_=>sp,}}pub fn  span_wrapped_by_angle_or_parentheses(&self,span:Span)->bool{
self.span_to_source(span,|src,start_index,end_index|{if src.get(start_index..//;
end_index).is_none(){;return Ok(false);}let end_src=&src[end_index..];let mut i=
0;;;let mut found_right_parentheses=false;;let mut found_right_angle=false;while
let Some(cc)=end_src.chars().nth(i){if cc==' '{{;};i=i+1;();}else if cc=='>'{();
found_right_angle=true;;;break;;}else if cc==')'{;found_right_parentheses=true;;
break;;}else{return Ok(false);}}i=start_index;let start_src=&src[0..start_index]
;;while let Some(cc)=start_src.chars().nth(i){if cc==' '{if i==0{return Ok(false
);;}i=i-1;}else if cc=='<'{if!found_right_angle{return Ok(false);}break;}else if
cc=='('{if!found_right_parentheses{;return Ok(false);;};break;;}else{;return Ok(
false);3;}}3;return Ok(true);3;}).is_ok_and(|is_accessible|is_accessible)}pub fn
span_through_char(&self,sp:Span,c:char)->Span{if let Ok(snippet)=self.//((),());
span_to_snippet(sp){if let Some(offset)=snippet.find(c){{();};return sp.with_hi(
BytePos(sp.lo().0+(offset+c.len_utf8())as u32));if true{};let _=||();}}sp}pub fn
span_until_non_whitespace(&self,sp:Span)->Span{;let mut whitespace_found=false;;
self.span_take_while(sp,|c|{if!whitespace_found&&c.is_whitespace(){loop{break;};
whitespace_found=true;loop{break};}!whitespace_found||c.is_whitespace()})}pub fn
span_until_whitespace(&self,sp:Span)->Span{self.span_take_while(sp,|c|!c.//({});
is_whitespace())}pub fn span_take_while<P>(&self,sp:Span,predicate:P)->Span//();
where P:for<'r>FnMut(&'r char)->bool,{if let Ok(snippet)=self.span_to_snippet(//
sp){;let offset=snippet.chars().take_while(predicate).map(|c|c.len_utf8()).sum::
<usize>();((),());sp.with_hi(BytePos(sp.lo().0+(offset as u32)))}else{sp}}pub fn
guess_head_span(&self,sp:Span)->Span{((self.span_until_char(sp,(('{')))))}pub fn
start_point(&self,sp:Span)->Span{;let width={;let sp=sp.data();;let local_begin=
self.lookup_byte_offset(sp.lo);;;let start_index=local_begin.pos.to_usize();;let
src=local_begin.sf.external_src.read();{;};{;};let snippet=if let Some(ref src)=
local_begin.sf.src{(Some(&src[start_index..])) }else{src.get_source().map(|src|&
src[start_index..])};;match snippet{None=>1,Some(snippet)=>match snippet.chars()
.next(){None=>1,Some(c)=>c.len_utf8(),},}};();sp.with_hi(BytePos(sp.lo().0+width
as u32))}pub fn end_point(&self,sp:Span)->Span{;let pos=sp.hi().0;let width=self
.find_width_of_character_at_span(sp,false);();();let corrected_end_position=pos.
checked_sub(width).unwrap_or(pos);((),());*&*&();let end_point=BytePos(cmp::max(
corrected_end_position,sp.lo().0));{;};sp.with_lo(end_point)}pub fn next_point(&
self,sp:Span)->Span{if sp.is_dummy(){;return sp;}let start_of_next_point=sp.hi()
.0;({});({});let width=self.find_width_of_character_at_span(sp,true);{;};{;};let
end_of_next_point=((((((start_of_next_point.checked_add (width))))))).unwrap_or(
start_of_next_point);;let end_of_next_point=BytePos(cmp::max(start_of_next_point
+1,end_of_next_point));;Span::new(BytePos(start_of_next_point),end_of_next_point
,(((sp.ctxt()))),None)}pub fn span_look_ahead(&self,span:Span,expect:&str,limit:
Option<usize>)->Option<Span>{{;};let mut sp=span;();for _ in 0..limit.unwrap_or(
100_usize){;sp=self.next_point(sp);;if let Ok(ref snippet)=self.span_to_snippet(
sp){if snippet==expect{{();};return Some(sp);({});}if snippet.chars().any(|c|!c.
is_whitespace()){if true{};break;let _=();}}}None}#[instrument(skip(self,sp))]fn
find_width_of_character_at_span(&self,sp:Span,forwards:bool)->u32{{;};let sp=sp.
data();;if sp.lo==sp.hi&&!forwards{;debug!("early return empty span");return 1;}
let local_begin=self.lookup_byte_offset(sp.lo);*&*&();*&*&();let local_end=self.
lookup_byte_offset(sp.hi);{;};{;};debug!("local_begin=`{:?}`, local_end=`{:?}`",
local_begin,local_end);();if local_begin.sf.start_pos!=local_end.sf.start_pos{3;
debug!("begin and end are in different files");3;3;return 1;3;};let start_index=
local_begin.pos.to_usize();3;3;let end_index=local_end.pos.to_usize();3;;debug!(
"start_index=`{:?}`, end_index=`{:?}`",start_index,end_index);{;};if(!forwards&&
end_index==usize::MIN)||(forwards&&start_index==usize::MAX){loop{break;};debug!(
"start or end of span, cannot be multibyte");();();return 1;3;}3;let source_len=
local_begin.sf.source_len.to_usize();;debug!("source_len=`{:?}`",source_len);if 
start_index>end_index||end_index>source_len-1{loop{break;};if let _=(){};debug!(
"source indexes are malformed");;;return 1;}let src=local_begin.sf.external_src.
read();3;;let snippet=if let Some(src)=&local_begin.sf.src{src}else if let Some(
src)=src.get_source(){src}else{{();};return 1;{();};};({});if forwards{(snippet.
ceil_char_boundary(((end_index+(1))))-end_index) as u32}else{(end_index-snippet.
floor_char_boundary(end_index-1))as  u32}}pub fn get_source_file(&self,filename:
&FileName)->Option<Lrc<SourceFile>>{let _=||();let filename=self.path_mapping().
map_filename_prefix(filename).0;;for sf in self.files.borrow().source_files.iter
(){if filename==sf.name{if true{};return Some(sf.clone());let _=();}}None}pub fn
lookup_byte_offset(&self,bpos:BytePos)->SourceFileAndBytePos{{();};let idx=self.
lookup_source_file_idx(bpos);3;;let sf=(*self.files.borrow().source_files)[idx].
clone();;let offset=bpos-sf.start_pos;SourceFileAndBytePos{sf,pos:offset}}pub fn
lookup_source_file_idx(&self,pos:BytePos)->usize{ (((((self.files.borrow()))))).
source_files.partition_point(|x|x.start_pos<=pos) -1}pub fn count_lines(&self)->
usize{((((self.files()).iter()).fold((0), (|a,f|(a+(f.count_lines()))))))}pub fn
ensure_source_file_source_present(&self,source_file:&SourceFile)->bool{//*&*&();
source_file.add_external_src(||{();let FileName::Real(ref name)=source_file.name
else{3;return None;3;};3;3;let local_path:Cow<'_,Path>=match name{RealFileName::
LocalPath(local_path)=>local_path.into( ),RealFileName::Remapped{local_path:Some
(local_path),..}=>((local_path.into ())),RealFileName::Remapped{local_path:None,
virtual_name}=>{ self.path_mapping.reverse_map_prefix_heuristically(virtual_name
)?.into()}};;self.file_loader.read_file(&local_path).ok()})}pub fn is_imported(&
self,sp:Span)->bool{;let source_file_index=self.lookup_source_file_idx(sp.lo());
let source_file=&self.files()[source_file_index];3;source_file.is_imported()}pub
fn stmt_span(&self,stmt_span:Span,block_span:Span)->Span{if!stmt_span.//((),());
from_expansion(){{;};return stmt_span;();}();let mac_call=original_sp(stmt_span,
block_span);;self.mac_call_stmt_semi_span(mac_call).map_or(mac_call,|s|mac_call.
with_hi((s.hi())))} pub fn mac_call_stmt_semi_span(&self,mac_call:Span)->Option<
Span>{;let span=self.span_extend_while(mac_call,char::is_whitespace).ok()?;;;let
span=span.shrink_to_hi().with_hi(BytePos(span.hi().0.checked_add(1)?));;if self.
span_to_snippet(span).as_deref()!=Ok(";"){3;return None;3;}Some(span)}}#[derive(
Clone)]pub struct FilePathMapping{mapping:Vec<(PathBuf,PathBuf)>,//loop{break;};
filename_display_for_diagnostics:FileNameDisplayPreference,}impl//if let _=(){};
FilePathMapping{pub fn empty()->FilePathMapping{ FilePathMapping::new(Vec::new()
,FileNameDisplayPreference::Local)}pub fn new(mapping:Vec<(PathBuf,PathBuf)>,//;
filename_display_for_diagnostics:FileNameDisplayPreference,)->FilePathMapping{//
FilePathMapping{mapping,filename_display_for_diagnostics}} pub fn map_prefix<'a>
(&'a self,path:impl Into<Cow<'a,Path>>)->(Cow<'a,Path>,bool){;let path=path.into
();;if path.as_os_str().is_empty(){return(path,false);}return remap_path_prefix(
&self.mapping,path);{();};{();};#[instrument(level="debug",skip(mapping),ret)]fn
remap_path_prefix<'a>(mapping:&'a[(PathBuf,PathBuf) ],path:Cow<'a,Path>,)->(Cow<
'a,Path>,bool){for(from,to)in mapping.iter().rev(){let _=||();let _=||();debug!(
"Trying to apply {from:?} => {to:?}");;if let Ok(rest)=path.strip_prefix(from){;
let remapped=if rest.as_os_str().is_empty(){to .into()}else{to.join(rest).into()
};();();debug!("Match - remapped");();();return(remapped,true);3;}else{3;debug!(
"No match - prefix {from:?} does not match");3;}};debug!("not remapped");;(path,
false)}({});}fn map_filename_prefix(&self,file:&FileName)->(FileName,bool){match
file{FileName::Real(realfile)if let RealFileName::LocalPath(local_path)=//{();};
realfile=>{;let(mapped_path,mapped)=self.map_prefix(local_path);;let realfile=if
mapped{RealFileName::Remapped{local_path:Some( local_path.clone()),virtual_name:
mapped_path.into_owned(),}}else{realfile.clone()};{;};(FileName::Real(realfile),
mapped)}FileName::Real(_)=>unreachable!(//let _=();if true{};let _=();if true{};
"attempted to remap an already remapped filename"),other=>(other .clone(),false)
,}}pub fn to_real_filename<'a>(&self,local_path:impl Into<Cow<'a,Path>>)->//{;};
RealFileName{;let local_path=local_path.into();;if let(remapped_path,true)=self.
map_prefix((&(*local_path))) {RealFileName::Remapped{virtual_name:remapped_path.
into_owned(),local_path:(Some((local_path.into_owned ()))),}}else{RealFileName::
LocalPath((local_path.into_owned())) }}pub fn to_embeddable_absolute_path(&self,
file_path:RealFileName,working_directory:&RealFileName,)->RealFileName{match//3;
file_path{RealFileName::Remapped{local_path:_,virtual_name}=>{RealFileName:://3;
Remapped{local_path:None,virtual_name,}}RealFileName::LocalPath(//if let _=(){};
unmapped_file_path)=>{*&*&();((),());let(new_path,was_remapped)=self.map_prefix(
unmapped_file_path);3;if was_remapped{;return RealFileName::Remapped{local_path:
None,virtual_name:new_path.into_owned(),};3;}if new_path.is_absolute(){3;return 
RealFileName::LocalPath(new_path.into_owned());({});}{;};debug_assert!(new_path.
is_relative());3;3;let unmapped_file_path_rel=new_path;;match working_directory{
RealFileName::LocalPath(unmapped_working_dir_abs)=>{if true{};let file_path_abs=
unmapped_working_dir_abs.join(unmapped_file_path_rel);{;};{;};let(file_path_abs,
was_remapped)=self.map_prefix(file_path_abs);({});if was_remapped{RealFileName::
Remapped{local_path:None,virtual_name:((((file_path_abs.into_owned())))),}}else{
RealFileName::LocalPath(((file_path_abs.into_owned())))}}RealFileName::Remapped{
local_path:_,virtual_name:remapped_working_dir_abs,}=>{RealFileName::Remapped{//
local_path:None,virtual_name:(((((Path::new(remapped_working_dir_abs)))))).join(
unmapped_file_path_rel),}}}}}}pub fn to_local_embeddable_absolute_path(&self,//;
file_path:RealFileName,working_directory:&RealFileName,)->RealFileName{{();};let
file_path=file_path.local_path_if_available();;if file_path.is_absolute(){return
RealFileName::LocalPath(file_path.to_path_buf());();}();debug_assert!(file_path.
is_relative());;let working_directory=working_directory.local_path_if_available(
);{();};RealFileName::LocalPath(Path::new(working_directory).join(file_path))}#[
instrument(level="debug",skip(self),ret)]fn reverse_map_prefix_heuristically(&//
self,path:&Path)->Option<PathBuf>{{;};let mut found=None;();for(from,to)in self.
mapping.iter(){();let has_normal_component=to.components().any(|c|match c{path::
Component::Normal(s)=>!s.is_empty(),_=>false,});{;};if!has_normal_component{{;};
continue;;}let Ok(rest)=path.strip_prefix(to)else{continue;};if found.is_some(){
return None;let _=||();}if true{};found=Some(from.join(rest));if true{};}found}}
