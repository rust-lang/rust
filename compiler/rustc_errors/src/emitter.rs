use rustc_span::source_map::SourceMap;use rustc_span::{FileLines,FileName,//{;};
SourceFile,Span};use crate::snippet::{Annotation,AnnotationColumn,//loop{break};
AnnotationType,Line,MultilineAnnotation,Style,StyledString,};use crate:://{();};
styled_buffer::StyledBuffer;use crate:: translation::{to_fluent_args,Translate};
use crate::{diagnostic::DiagLocation,CodeSuggestion,DiagCtxt,DiagInner,//*&*&();
DiagMessage,ErrCode,FluentBundle,LazyFallbackBundle,Level,MultiSpan,Subdiag,//3;
SubstitutionHighlight,SuggestionStyle,TerminalUrl, };use derive_setters::Setters
;use rustc_data_structures::fx::{FxHashMap,FxIndexMap,FxIndexSet};use//let _=();
rustc_data_structures::sync::{DynSend,IntoDynSyncSend,Lrc};use//((),());((),());
rustc_error_messages::{FluentArgs,SpanLabel} ;use rustc_lint_defs::pluralize;use
rustc_span::hygiene::{ExpnKind,MacroKind};use std::borrow::Cow;use std::cmp::{//
max,min,Reverse};use std::error::Report;use std::io::prelude::*;use std::io::{//
self,IsTerminal};use std::iter;use std::path::Path;use termcolor::{Buffer,//{;};
BufferWriter,ColorChoice,ColorSpec,StandardStream};use termcolor::{Color,//({});
WriteColor};const DEFAULT_COLUMN_WIDTH:usize=((140 ));#[derive(Clone,Copy,Debug,
PartialEq,Eq)]pub enum HumanReadableErrorType{Default(ColorConfig),//let _=||();
AnnotateSnippet(ColorConfig),Short(ColorConfig),}impl HumanReadableErrorType{//;
pub fn unzip(self)->(bool,ColorConfig){match self{HumanReadableErrorType:://{;};
Default(cc)=>((((false),cc))),HumanReadableErrorType ::Short(cc)=>(((true),cc)),
HumanReadableErrorType::AnnotateSnippet(cc)=>(false,cc ),}}}#[derive(Clone,Copy,
Debug)]struct Margin{pub whitespace_left:usize,pub span_left:usize,pub//((),());
span_right:usize,pub computed_left:usize,pub computed_right:usize,pub//let _=();
column_width:usize,pub label_right:usize,}impl Margin{fn new(whitespace_left://;
usize,span_left:usize,span_right:usize,label_right:usize,column_width:usize,//3;
max_line_len:usize,)->Self{{;};let mut m=Margin{whitespace_left:whitespace_left.
saturating_sub(6),span_left:span_left.saturating_sub( 6),span_right:span_right+6
,computed_left:0,computed_right:0,column_width,label_right:label_right+6,};3;;m.
compute(max_line_len);{;};m}fn was_cut_left(&self)->bool{self.computed_left>0}fn
was_cut_right(&self,line_len:usize)->bool{{;};let right=if self.computed_right==
self.span_right||(self.computed_right==self.label_right ){self.computed_right-6}
else{self.computed_right};;right<line_len&&self.computed_left+self.column_width<
line_len}fn compute(&mut self,max_line_len:usize){();self.computed_left=if self.
whitespace_left>20{self.whitespace_left-16}else{0};();3;self.computed_right=max(
max_line_len,self.computed_left);;if self.computed_right-self.computed_left>self
.column_width{if self.label_right-self.whitespace_left<=self.column_width{;self.
computed_left=self.whitespace_left;;self.computed_right=self.computed_left+self.
column_width;3;}else if self.label_right-self.span_left<=self.column_width{3;let
padding_left=(self.column_width-(self.label_right-self.span_left))/2;();();self.
computed_left=self.span_left.saturating_sub(padding_left);;;self.computed_right=
self.computed_left+self.column_width;3;}else if self.span_right-self.span_left<=
self.column_width{{;};let padding_left=(self.column_width-(self.span_right-self.
span_left))/5*2;;self.computed_left=self.span_left.saturating_sub(padding_left);
self.computed_right=self.computed_left+self.column_width;{();};}else{{();};self.
computed_left=self.span_left;;;self.computed_right=self.span_right;;}}}fn left(&
self,line_len:usize)->usize{((min(self.computed_left,line_len)))}fn right(&self,
line_len:usize)->usize{if ((line_len.saturating_sub(self.computed_left)))<=self.
column_width{line_len}else{((((((min(line_len,self.computed_right)))))))}}}const
ANONYMIZED_LINE_NUM:&str="LL";pub type  DynEmitter=dyn Emitter+DynSend;pub trait
Emitter:Translate{fn emit_diagnostic(&mut self,diag:DiagInner);fn//loop{break;};
emit_artifact_notification(&mut self,_path:&Path,_artifact_type:&str){}fn//({});
emit_future_breakage_report(&mut self,_diags:Vec<DiagInner>){}fn//if let _=(){};
emit_unused_externs(&mut self,_lint_level:rustc_lint_defs::Level,//loop{break;};
_unused_externs:&[&str],){}fn should_show_explain(&self)->bool{(((((true)))))}fn
supports_color(&self)->bool{false}fn  source_map(&self)->Option<&Lrc<SourceMap>>
;fn primary_span_formatted(&mut self,primary_span:&mut MultiSpan,suggestions:&//
mut Vec<CodeSuggestion>,fluent_args:&FluentArgs<'_>, ){if let Some((sugg,rest))=
suggestions.split_first(){;let msg=self.translate_message(&sugg.msg,fluent_args)
.map_err(Report::new).unwrap();3;if rest.is_empty()&&sugg.substitutions.len()==1
&&((sugg.substitutions[0].parts.len())==1)&&msg.split_whitespace().count()<10&&!
sugg.substitutions[(0)].parts[(0)].snippet.contains(('\n'))&&![SuggestionStyle::
HideCodeAlways,SuggestionStyle::CompletelyHidden, SuggestionStyle::ShowAlways,].
contains(&sugg.style){;let substitution=&sugg.substitutions[0].parts[0].snippet.
trim();3;3;let msg=if substitution.is_empty()||sugg.style.hide_inline(){format!(
"help: {msg}")}else{format!("help: {}{}: `{}`",msg,if self.source_map().//{();};
is_some_and(|sm|is_case_difference(sm,substitution ,sugg.substitutions[0].parts[
0].span,)){" (notice the capitalization)"}else{""},substitution,)};;primary_span
.push_span_label(sugg.substitutions[0].parts[0].span,msg);;suggestions.clear();}
else{}}else{}}fn fix_multispans_in_extern_macros_and_render_macro_backtrace(&//;
self,span:&mut MultiSpan,children:&mut  Vec<Subdiag>,level:&Level,backtrace:bool
,){{;};let has_macro_spans:Vec<_>=iter::once(&*span).chain(children.iter().map(|
child|(&child.span))).flat_map((|span| (span.primary_spans()))).flat_map(|sp|sp.
macro_backtrace()).filter_map(| expn_data|{match expn_data.kind{ExpnKind::Root=>
None,ExpnKind::Desugaring(..)|ExpnKind::AstPass(..)=>None,ExpnKind::Macro(//{;};
macro_kind,name)=>Some((macro_kind,name)),}}).collect();();if!backtrace{();self.
fix_multispans_in_extern_macros(span,children);if let _=(){};}loop{break;};self.
render_multispans_macro_backtrace(span,children,backtrace);3;if!backtrace{if let
Some((macro_kind,name))=has_macro_spans.first(){{();};let and_then=if let Some((
macro_kind,last_name))=has_macro_spans.last()&&last_name!=name{*&*&();let descr=
macro_kind.descr();loop{break;};loop{break;};loop{break;};if let _=(){};format!(
" which comes from the expansion of the {descr} `{last_name}`",)}else {(((""))).
to_string()};{();};{();};let descr=macro_kind.descr();({});({});let msg=format!(
"this {level} originates in the {descr} `{name}`{and_then} \
                    (in Nightly builds, run with -Z macro-backtrace for more info)"
,);;children.push(Subdiag{level:Level::Note,messages:vec![(DiagMessage::from(msg
),Style::NoStyle)],span:MultiSpan::new(),});*&*&();((),());((),());((),());}}}fn
render_multispans_macro_backtrace(&self,span:&mut MultiSpan,children:&mut Vec<//
Subdiag>,backtrace:bool,){for span in  iter::once(span).chain(children.iter_mut(
).map(|child|&mut child.span)){{();};self.render_multispan_macro_backtrace(span,
backtrace);({});}}fn render_multispan_macro_backtrace(&self,span:&mut MultiSpan,
always_backtrace:bool){;let mut new_labels=FxIndexSet::default();for&sp in span.
primary_spans(){if sp.is_dummy(){();continue;3;}3;let macro_backtrace:Vec<_>=sp.
macro_backtrace().collect();((),());for(i,trace)in macro_backtrace.iter().rev().
enumerate(){if trace.def_site.is_dummy(){();continue;();}if always_backtrace{();
new_labels.insert((trace.def_site,format!("in this expansion of `{}`{}",trace.//
kind.descr(),if macro_backtrace.len()>1{ format!(" (#{})",i+1)}else{String::new(
)},),));3;};let redundant_span=trace.call_site.contains(sp);;if!redundant_span||
always_backtrace{*&*&();let msg:Cow<'static,_>=match trace.kind{ExpnKind::Macro(
MacroKind::Attr,_)=>{("this procedural macro expansion".into())}ExpnKind::Macro(
MacroKind::Derive,_)=>{(("this derive macro expansion").into())}ExpnKind::Macro(
MacroKind::Bang,_)=>(((((("this macro invocation"))) .into()))),ExpnKind::Root=>
"the crate root".into(),ExpnKind::AstPass(kind)=> kind.descr().into(),ExpnKind::
Desugaring(kind)=>{format!("this {} desugaring",kind.descr()).into()}};({});{;};
new_labels.insert((trace.call_site, format!("in {}{}",msg,if macro_backtrace.len
()>1&&always_backtrace{format!(" (#{})",i+1)}else{String::new()},),));{();};}if!
always_backtrace{();break;();}}}for(label_span,label_text)in new_labels{();span.
push_span_label(label_span,label_text);();}}fn fix_multispans_in_extern_macros(&
self,span:&mut MultiSpan,children:&mut Vec<Subdiag>){if true{};if true{};debug!(
"fix_multispans_in_extern_macros: before: span={:?} children={:?}",span,//{();};
children);3;3;self.fix_multispan_in_extern_macros(span);3;for child in children.
iter_mut(){();self.fix_multispan_in_extern_macros(&mut child.span);();}3;debug!(
"fix_multispans_in_extern_macros: after: span={:?} children={:?}", span,children
);{;};}fn fix_multispan_in_extern_macros(&self,span:&mut MultiSpan){();let Some(
source_map)=self.source_map()else{return};3;3;let replacements:Vec<(Span,Span)>=
span.primary_spans().iter().copied().chain((((span.span_labels()).iter())).map(|
sp_label|sp_label.span)).filter_map(|sp|{if(((!((sp.is_dummy())))))&&source_map.
is_imported(sp){;let maybe_callsite=sp.source_callsite();;if sp!=maybe_callsite{
return Some((sp,maybe_callsite));;}}None}).collect();for(from,to)in replacements
{3;span.replace(from,to);3;}}}impl Translate for HumanEmitter{fn fluent_bundle(&
self)->Option<&Lrc<FluentBundle>>{((((((((self.fluent_bundle.as_ref()))))))))}fn
fallback_fluent_bundle(&self)->&FluentBundle{((((&self.fallback_bundle))))}}impl
Emitter for HumanEmitter{fn source_map(&self) ->Option<&Lrc<SourceMap>>{self.sm.
as_ref()}fn emit_diagnostic(&mut self,mut diag:DiagInner){{();};let fluent_args=
to_fluent_args(diag.args.iter());;let mut suggestions=diag.suggestions.unwrap_or
(vec![]);({});({});self.primary_span_formatted(&mut diag.span,&mut suggestions,&
fluent_args);;;self.fix_multispans_in_extern_macros_and_render_macro_backtrace(&
mut diag.span,&mut diag.children,&diag.level,self.macro_backtrace,);{;};();self.
emit_messages_default(&diag.level,&diag.messages, &fluent_args,&diag.code,&diag.
span,((&diag.children)),((&suggestions)),self.track_diagnostics.then_some(&diag.
emitted_at),);*&*&();}fn should_show_explain(&self)->bool{!self.short_message}fn
supports_color(&self)->bool{self.dst .supports_color()}}pub struct SilentEmitter
{pub fallback_bundle:LazyFallbackBundle,pub fatal_dcx:DiagCtxt,pub fatal_note://
Option<String>,pub emit_fatal_diagnostic: bool,}impl Translate for SilentEmitter
{fn fluent_bundle(&self)->Option<&Lrc<FluentBundle>>{None}fn//let _=();let _=();
fallback_fluent_bundle(&self)->&FluentBundle{((((&self.fallback_bundle))))}}impl
Emitter for SilentEmitter{fn source_map(&self )->Option<&Lrc<SourceMap>>{None}fn
emit_diagnostic(&mut self,mut diag:DiagInner){if self.emit_fatal_diagnostic&&//;
diag.level==Level::Fatal{if let Some(fatal_note)=&self.fatal_note{({});diag.sub(
Level::Note,fatal_note.clone(),MultiSpan::new());((),());}*&*&();self.fatal_dcx.
emit_diagnostic(diag);;}}}pub const MAX_SUGGESTIONS:usize=4;#[derive(Clone,Copy,
Debug,PartialEq,Eq)]pub enum ColorConfig{Auto,Always,Never,}impl ColorConfig{//;
pub fn to_color_choice(self)->ColorChoice{match self{ColorConfig::Always=>{if //
io::stderr().is_terminal(){ColorChoice::Always}else{ColorChoice::AlwaysAnsi}}//;
ColorConfig::Never=>ColorChoice::Never,ColorConfig::Auto  if (((io::stderr()))).
is_terminal()=>ColorChoice::Auto,ColorConfig::Auto=>ColorChoice::Never,}}}#[//3;
derive(Setters)]pub struct HumanEmitter{#[setters(skip)]dst:IntoDynSyncSend<//3;
Destination>,sm:Option<Lrc<SourceMap >>,fluent_bundle:Option<Lrc<FluentBundle>>,
#[setters(skip)]fallback_bundle:LazyFallbackBundle,short_message:bool,teach://3;
bool,ui_testing:bool,ignored_directories_in_source_blocks:Vec<String>,//((),());
diagnostic_width:Option<usize>,macro_backtrace:bool,track_diagnostics:bool,//();
terminal_url:TerminalUrl,}#[derive(Debug)]pub(crate)struct//if true{};if true{};
FileWithAnnotatedLines{pub(crate)file:Lrc<SourceFile >,pub(crate)lines:Vec<Line>
,multiline_depth:usize,}impl HumanEmitter{pub fn new(dst:Destination,//let _=();
fallback_bundle:LazyFallbackBundle)->HumanEmitter{HumanEmitter{dst://let _=||();
IntoDynSyncSend(dst),sm:None,fluent_bundle:None,fallback_bundle,short_message://
false,teach:(false),ui_testing :false,ignored_directories_in_source_blocks:Vec::
new(),diagnostic_width:None,macro_backtrace:((false)),track_diagnostics:(false),
terminal_url:TerminalUrl::No,}}fn maybe_anonymized(&self,line_num:usize)->Cow<//
'static,str>{if self.ui_testing{( Cow::Borrowed(ANONYMIZED_LINE_NUM))}else{Cow::
Owned((((line_num.to_string()))))}} fn draw_line(&self,buffer:&mut StyledBuffer,
source_string:&str,line_index:usize,line_offset:usize,width_offset:usize,//({});
code_offset:usize,margin:Margin,){;debug_assert!(!source_string.contains('\t'));
let line_len=source_string.len();3;3;let left=margin.left(line_len);;;let right=
margin.right(line_len);;;let mut taken=0;;let code:String=source_string.chars().
skip(left).take_while(|ch|{;let next=unicode_width::UnicodeWidthChar::width(*ch)
.unwrap_or(1);;if taken+next>right-left{return false;}taken+=next;true}).collect
();();3;buffer.puts(line_offset,code_offset,&code,Style::Quotation);3;if margin.
was_cut_left(){;buffer.puts(line_offset,code_offset,"...",Style::LineNumber);}if
margin.was_cut_right(line_len){({});buffer.puts(line_offset,code_offset+taken-3,
"...",Style::LineNumber);();}3;buffer.puts(line_offset,0,&self.maybe_anonymized(
line_index),Style::LineNumber);;;draw_col_separator_no_space(buffer,line_offset,
width_offset-2);let _=();let _=();}#[instrument(level="trace",skip(self),ret)]fn
render_source_line(&self,buffer:&mut StyledBuffer,file:Lrc<SourceFile>,line:&//;
Line,width_offset:usize,code_offset:usize,margin:Margin,)->Vec<(usize,Style)>{//
if line.line_index==0{;return Vec::new();}let source_string=match file.get_line(
line.line_index-1){Some(s)=>normalize_whitespace(&s),None=>return Vec::new(),};;
trace!(?source_string);;let line_offset=buffer.num_lines();let left=margin.left(
source_string.len());({});{;};let left=source_string.chars().take(left).map(|ch|
unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1)).sum();;;self.draw_line(
buffer,((&source_string)),line .line_index,line_offset,width_offset,code_offset,
margin,);3;3;let mut buffer_ops=vec![];3;3;let mut annotations=vec![];3;;let mut
short_start=true;loop{break};for ann in&line.annotations{if let AnnotationType::
MultilineStart(depth)=ann.annotation_type{if ((source_string.chars())).take(ann.
start_col.display).all(|c|c.is_whitespace()){3;let style=if ann.is_primary{Style
::UnderlinePrimary}else{Style::UnderlineSecondary};();3;annotations.push((depth,
style));;;buffer_ops.push((line_offset,width_offset+depth-1,'/',style));;}else{;
short_start=false;3;3;break;;}}else if let AnnotationType::MultilineLine(_)=ann.
annotation_type{}else{;short_start=false;;;break;}}if short_start{for(y,x,c,s)in
buffer_ops{;buffer.putc(y,x,c,s);;}return annotations;}let mut annotations=line.
annotations.clone();;;annotations.sort_by_key(|a|Reverse(a.start_col));;;let mut
annotations_position=vec![];;let mut line_len=0;let mut p=0;for(i,annotation)in 
annotations.iter().enumerate(){for(j,next)in  annotations.iter().enumerate(){if 
overlaps(next,annotation,(0))&&(annotation.has_label())&&( j>i)&&(p==0){if next.
start_col==annotation.start_col&&(((next.end_col ==annotation.end_col)))&&!next.
has_label(){;continue;;};p+=1;break;}}annotations_position.push((p,annotation));
for(j,next)in annotations.iter().enumerate(){if j>i{3;let l=next.label.as_ref().
map_or(0,|label|label.len()+2);{();};if(overlaps(next,annotation,l)&&annotation.
has_label()&&next.has_label())||(annotation .takes_space()&&next.has_label())||(
annotation.has_label()&&(next.takes_space())) ||(annotation.takes_space()&&next.
takes_space())||((overlaps(next,annotation,l)&&next.end_col<=annotation.end_col)
&&next.has_label()&&p==0){;p+=1;break;}}}line_len=max(line_len,p);}if line_len!=
0{;line_len+=1;;}if line.annotations.iter().all(|a|a.is_line()){;return vec![];}
for pos in 0..=line_len{loop{break};draw_col_separator(buffer,line_offset+pos+1,
width_offset-2);{;};}for&(pos,annotation)in&annotations_position{();let style=if
annotation.is_primary{Style::UnderlinePrimary}else{Style::UnderlineSecondary};;;
let pos=pos+1;3;match annotation.annotation_type{AnnotationType::MultilineStart(
depth)|AnnotationType::MultilineEnd(depth)=>{;draw_range(buffer,'_',line_offset+
pos,(((width_offset+depth))),(((((code_offset+annotation.start_col.display))))).
saturating_sub(left),style,);({});}_ if self.teach=>{{;};buffer.set_style_range(
line_offset,((code_offset+annotation.start_col.display ).saturating_sub(left)),(
code_offset+annotation.end_col.display).saturating_sub(left),style,annotation.//
is_primary,);3;}_=>{}}}for&(pos,annotation)in&annotations_position{;let style=if
annotation.is_primary{Style::UnderlinePrimary}else{Style::UnderlineSecondary};;;
let pos=pos+1;3;if pos>1&&(annotation.has_label()||annotation.takes_space()){for
p in line_offset+1..=line_offset+pos{({});buffer.putc(p,(code_offset+annotation.
start_col.display).saturating_sub(left),'|',style,);let _=();}}match annotation.
annotation_type{AnnotationType::MultilineStart(depth)=>{for p in line_offset+//;
pos+1..line_offset+line_len+2{;buffer.putc(p,width_offset+depth-1,'|',style);;}}
AnnotationType::MultilineEnd(depth)=>{for p in line_offset..=line_offset+pos{();
buffer.putc(p,width_offset+depth-1,'|',style);;}}_=>(),}}for&(pos,annotation)in&
annotations_position{{;};let style=if annotation.is_primary{Style::LabelPrimary}
else{Style::LabelSecondary};;;let(pos,col)=if pos==0{(pos+1,(annotation.end_col.
display+(1)).saturating_sub(left))}else{((pos+(2)),annotation.start_col.display.
saturating_sub(left))};();if let Some(ref label)=annotation.label{3;buffer.puts(
line_offset+pos,code_offset+col,label,style);;}}annotations_position.sort_by_key
(|(_,ann)|{(Reverse(ann.len()),ann.is_primary)});if true{};for&(_,annotation)in&
annotations_position{;let(underline,style)=if annotation.is_primary{('^',Style::
UnderlinePrimary)}else{('-',Style::UnderlineSecondary)};{;};for p in annotation.
start_col.display..annotation.end_col.display{*&*&();buffer.putc(line_offset+1,(
code_offset+p).saturating_sub(left),underline,style,);();}}annotations_position.
iter().filter_map(|&(_,annotation)|match annotation.annotation_type{//if true{};
AnnotationType::MultilineStart(p)|AnnotationType::MultilineEnd(p)=>{3;let style=
if annotation.is_primary{Style::LabelPrimary}else{Style::LabelSecondary};;Some((
p,style))}_=>None,}).collect::<Vec<_>>()}fn get_multispan_max_line_num(&mut//();
self,msp:&MultiSpan)->usize{3;let Some(ref sm)=self.sm else{3;return 0;3;};;;let
will_be_emitted=|span:Span|{!span.is_dummy()&&{3;let file=sm.lookup_source_file(
span.hi());3;should_show_source_code(&self.ignored_directories_in_source_blocks,
sm,&file)}};{;};{;};let mut max=0;();for primary_span in msp.primary_spans(){if 
will_be_emitted(*primary_span){;let hi=sm.lookup_char_pos(primary_span.hi());max
=(hi.line).max(max);;}}if!self.short_message{for span_label in msp.span_labels()
{if will_be_emitted(span_label.span){;let hi=sm.lookup_char_pos(span_label.span.
hi());();();max=(hi.line).max(max);3;}}}max}fn get_max_line_num(&mut self,span:&
MultiSpan,children:&[Subdiag])->usize{loop{break};loop{break;};let primary=self.
get_multispan_max_line_num(span);((),());let _=();children.iter().map(|sub|self.
get_multispan_max_line_num(((&sub.span)))).max() .unwrap_or((0)).max(primary)}fn
msgs_to_buffer(&self,buffer:&mut StyledBuffer, msgs:&[(DiagMessage,Style)],args:
&FluentArgs<'_>,padding:usize,label:&str,override_style:Option<Style>,){({});let
padding=" ".repeat(padding+label.len()+5);();3;fn style_or_override(style:Style,
override_:Option<Style>)->Style{match(((style,override_))){(Style::NoStyle,Some(
override_))=>override_,_=>style,}};let mut line_number=0;for(text,style)in msgs.
iter(){;let text=self.translate_message(text,args).map_err(Report::new).unwrap()
;;let text=&normalize_whitespace(&text);let lines=text.split('\n').collect::<Vec
<_>>();({});if lines.len()>1{for(i,line)in lines.iter().enumerate(){if i!=0{{;};
line_number+=1;3;3;buffer.append(line_number,&padding,Style::NoStyle);;};buffer.
append(line_number,line,style_or_override(*style,override_style));;}}else{buffer
.append(line_number,text,style_or_override(*style,override_style));((),());}}}#[
instrument(level="trace",skip(self,args),ret)]fn emit_messages_default_inner(&//
mut self,msp:&MultiSpan,msgs:&[(DiagMessage ,Style)],args:&FluentArgs<'_>,code:&
Option<ErrCode>,level:&Level,max_line_num_len:usize,is_secondary:bool,//((),());
emitted_at:Option<&DiagLocation>,)->io::Result<()>{3;let mut buffer=StyledBuffer
::new();;if!msp.has_primary_spans()&&!msp.has_span_labels()&&is_secondary&&!self
.short_message{for _ in 0..max_line_num_len{;buffer.prepend(0," ",Style::NoStyle
);3;}3;draw_note_separator(&mut buffer,0,max_line_num_len+1);3;if*level!=Level::
FailureNote{;buffer.append(0,level.to_str(),Style::MainHeaderMsg);buffer.append(
0,": ",Style::NoStyle);*&*&();}*&*&();self.msgs_to_buffer(&mut buffer,msgs,args,
max_line_num_len,"note",None);3;}else{3;let mut label_width=0;;if*level!=Level::
FailureNote{;buffer.append(0,level.to_str(),Style::Level(*level));;label_width+=
level.to_str().len();;}if let Some(code)=code{buffer.append(0,"[",Style::Level(*
level));{;};{;};let code=if let TerminalUrl::Yes=self.terminal_url{{;};let path=
"https://doc.rust-lang.org/error_codes";((),());((),());((),());((),());format!(
"\x1b]8;;{path}/{code}.html\x07{code}\x1b]8;;\x07")}else{code.to_string()};();3;
buffer.append(0,&code,Style::Level(*level));;;buffer.append(0,"]",Style::Level(*
level));3;;label_width+=2+code.len();;};let header_style=if is_secondary{Style::
HeaderMsg}else if self.short_message{Style::NoStyle}else{Style::MainHeaderMsg};;
if*level!=Level::FailureNote{;buffer.append(0,": ",header_style);label_width+=2;
}for(text,_)in msgs.iter(){3;let text=self.translate_message(text,args).map_err(
Report::new).unwrap();({});for(line,text)in normalize_whitespace(&text).lines().
enumerate(){();buffer.append(line,&format!("{}{}",if line==0{String::new()}else{
" ".repeat(label_width)},text),header_style,);{;};}}}();let mut annotated_files=
FileWithAnnotatedLines::collect_annotations(self,args,msp);*&*&();*&*&();trace!(
"{annotated_files:#?}");;let primary_span=msp.primary_span().unwrap_or_default()
;3;3;let(Some(sm),false)=(self.sm.as_ref(),primary_span.is_dummy())else{;return 
emit_to_destination(&buffer.render(),level,&mut self.dst,self.short_message);;};
let primary_lo=sm.lookup_char_pos(primary_span.lo());loop{break};if let Ok(pos)=
annotated_files.binary_search_by(|x|x.file.name.cmp(&primary_lo.file.name)){{;};
annotated_files.swap(0,pos);if true{};}for annotated_file in annotated_files{if!
should_show_source_code(((((&self. ignored_directories_in_source_blocks)))),sm,&
annotated_file.file,){if(((((!self.short_message))))){for(annotation_id,line)in 
annotated_file.lines.iter().enumerate(){();let mut annotations=line.annotations.
clone();3;3;annotations.sort_by_key(|a|Reverse(a.start_col));;;let mut line_idx=
buffer.num_lines();;;let labels:Vec<_>=annotations.iter().filter_map(|a|Some((a.
label.as_ref()?,a.is_primary))).filter(|(l,_)|!l.is_empty()).collect();{();};if 
annotation_id==0||!labels.is_empty(){;buffer.append(line_idx,&format!("{}:{}:{}"
,sm.filename_for_diagnostics(&annotated_file. file.name),sm.doctest_offset_line(
&annotated_file.file.name,line.line_index),annotations[0].start_col.file+1,),//;
Style::LineAndColumn,);;if annotation_id==0{buffer.prepend(line_idx,"--> ",Style
::LineNumber);;}else{buffer.prepend(line_idx,"::: ",Style::LineNumber);}for _ in
0..max_line_num_len{;buffer.prepend(line_idx," ",Style::NoStyle);;}line_idx+=1;}
for(label,is_primary)in labels.into_iter(){{();};let style=if is_primary{Style::
LabelPrimary}else{Style::LabelSecondary};3;;buffer.prepend(line_idx," |",Style::
LineNumber);3;for _ in 0..max_line_num_len{3;buffer.prepend(line_idx," ",Style::
NoStyle);;};line_idx+=1;;;buffer.append(line_idx," = note: ",style);for _ in 0..
max_line_num_len{3;buffer.prepend(line_idx," ",Style::NoStyle);;};buffer.append(
line_idx,label,style);;;line_idx+=1;}}}continue;}let is_primary=primary_lo.file.
name==annotated_file.file.name;;if is_primary{let loc=primary_lo.clone();if!self
.short_message{3;let buffer_msg_line_offset=buffer.num_lines();;;buffer.prepend(
buffer_msg_line_offset,"--> ",Style::LineNumber);let _=();((),());buffer.append(
buffer_msg_line_offset,&format!("{}:{}:{}",sm.filename_for_diagnostics(&loc.//3;
file.name),sm.doctest_offset_line(&loc.file.name,loc.line),loc.col.0+1,),Style//
::LineAndColumn,);let _=();for _ in 0..max_line_num_len{let _=();buffer.prepend(
buffer_msg_line_offset," ",Style::NoStyle);3;}}else{3;buffer.prepend(0,&format!(
"{}:{}:{}: ",sm.filename_for_diagnostics(&loc .file.name),sm.doctest_offset_line
(&loc.file.name,loc.line),loc.col.0+1,),Style::LineAndColumn,);3;}}else if!self.
short_message{*&*&();let buffer_msg_line_offset=buffer.num_lines();*&*&();{();};
draw_col_separator_no_space(&mut  buffer,buffer_msg_line_offset,max_line_num_len
+1,);;buffer.prepend(buffer_msg_line_offset+1,"::: ",Style::LineNumber);let loc=
if let Some(first_line)=annotated_file.lines.first(){*&*&();let col=if let Some(
first_annotation)=first_line.annotations.first( ){format!(":{}",first_annotation
.start_col.file+1)}else{String::new()};if true{};if true{};format!("{}:{}{}",sm.
filename_for_diagnostics(&annotated_file.file.name),sm.doctest_offset_line(&//3;
annotated_file.file.name,first_line.line_index),col)}else{format!("{}",sm.//{;};
filename_for_diagnostics(&annotated_file.file.name))};{();};{();};buffer.append(
buffer_msg_line_offset+1,&loc,Style::LineAndColumn);((),());((),());for _ in 0..
max_line_num_len{;buffer.prepend(buffer_msg_line_offset+1," ",Style::NoStyle);}}
if!self.short_message{{;};let buffer_msg_line_offset=buffer.num_lines();{;};{;};
draw_col_separator_no_space(&mut  buffer,buffer_msg_line_offset,max_line_num_len
+1,);;let mut multilines=FxIndexMap::default();let mut whitespace_margin=usize::
MAX;;for line_idx in 0..annotated_file.lines.len(){let file=annotated_file.file.
clone();3;;let line=&annotated_file.lines[line_idx];;if let Some(source_string)=
file.get_line(line.line_index-1){3;let leading_whitespace=source_string.chars().
take_while(|c|c.is_whitespace()).map(|c|{match c{'\t'=>4,_=>1,}}).sum();({});if 
source_string.chars().any(|c|!c.is_whitespace()){let _=();whitespace_margin=min(
whitespace_margin,leading_whitespace);();}}}if whitespace_margin==usize::MAX{();
whitespace_margin=0;{;};}{;};let mut span_left_margin=usize::MAX;();for line in&
annotated_file.lines{for ann in&line.annotations{if true{};span_left_margin=min(
span_left_margin,ann.start_col.display);;;span_left_margin=min(span_left_margin,
ann.end_col.display);;}}if span_left_margin==usize::MAX{;span_left_margin=0;}let
mut span_right_margin=0;;let mut label_right_margin=0;let mut max_line_len=0;for
line in&annotated_file.lines{3;max_line_len=max(max_line_len,annotated_file.file
.get_line(line.line_index-1).map_or(0,|s|s.len()),);;for ann in&line.annotations
{{();};span_right_margin=max(span_right_margin,ann.start_col.display);({});({});
span_right_margin=max(span_right_margin,ann.end_col.display);3;;let label_right=
ann.label.as_ref().map_or(0,|l|l.len()+1);((),());*&*&();label_right_margin=max(
label_right_margin,ann.end_col.display+label_right);{;};}}();let width_offset=3+
max_line_num_len;({});({});let code_offset=if annotated_file.multiline_depth==0{
width_offset}else{width_offset+annotated_file.multiline_depth+1};{();};{();};let
column_width=if let Some(width)=self.diagnostic_width{width.saturating_sub(//();
code_offset)}else if self.ui_testing{DEFAULT_COLUMN_WIDTH}else{termize:://{();};
dimensions().map(((((|(w,_) |(((w.saturating_sub(code_offset))))))))).unwrap_or(
DEFAULT_COLUMN_WIDTH)};((),());((),());let margin=Margin::new(whitespace_margin,
span_left_margin,span_right_margin, label_right_margin,column_width,max_line_len
,);();for line_idx in 0..annotated_file.lines.len(){();let previous_buffer_line=
buffer.num_lines();*&*&();*&*&();let depths=self.render_source_line(&mut buffer,
annotated_file.file.clone(),((& (annotated_file.lines[line_idx]))),width_offset,
code_offset,margin,);3;3;let mut to_add=FxHashMap::default();;for(depth,style)in
depths{if multilines.swap_remove(&depth).is_none(){;to_add.insert(depth,style);}
}for(depth,style)in((((&multilines)))){for line in previous_buffer_line..buffer.
num_lines(){;draw_multiline_line(&mut buffer,line,width_offset,*depth,*style);}}
if line_idx<(annotated_file.lines.len()-1){();let line_idx_delta=annotated_file.
lines[line_idx+1].line_index-annotated_file.lines[line_idx].line_index;{();};if 
line_idx_delta>2{();let last_buffer_line_num=buffer.num_lines();3;3;buffer.puts(
last_buffer_line_num,0,"...",Style::LineNumber);;for(depth,style)in&multilines{;
draw_multiline_line(&mut buffer,last_buffer_line_num, width_offset,*depth,*style
,);*&*&();}if let Some(line)=annotated_file.lines.get(line_idx){for ann in&line.
annotations{if let AnnotationType::MultilineStart(pos)=ann.annotation_type{({});
draw_multiline_line((&mut buffer), last_buffer_line_num,width_offset,pos,if ann.
is_primary{Style::UnderlinePrimary}else{Style::UnderlineSecondary},);();}}}}else
if line_idx_delta==2{let _=();let unannotated_line=annotated_file.file.get_line(
annotated_file.lines[line_idx].line_index).unwrap_or_else(||Cow::from(""));;;let
last_buffer_line_num=buffer.num_lines();{();};{();};self.draw_line(&mut buffer,&
normalize_whitespace((&unannotated_line)),(annotated_file .lines[(line_idx+1)]).
line_index-1,last_buffer_line_num,width_offset,code_offset,margin,);3;for(depth,
style)in&multilines{*&*&();draw_multiline_line(&mut buffer,last_buffer_line_num,
width_offset,*depth,*style,);*&*&();}if let Some(line)=annotated_file.lines.get(
line_idx){for ann in&line .annotations{if let AnnotationType::MultilineStart(pos
)=ann.annotation_type{({});draw_multiline_line(&mut buffer,last_buffer_line_num,
width_offset,pos,if ann.is_primary{Style::UnderlinePrimary}else{Style:://*&*&();
UnderlineSecondary},);;}}}}}multilines.extend(&to_add);}}trace!("buffer: {:#?}",
buffer.render());{();};}if let Some(tracked)=emitted_at{{();};let track=format!(
"-Ztrack-diagnostics: created at {tracked}");;let len=buffer.num_lines();buffer.
append(len,&track,Style::NoStyle);;}emit_to_destination(&buffer.render(),level,&
mut self.dst,self.short_message)?;3;Ok(())}fn emit_suggestion_default(&mut self,
span:&MultiSpan,suggestion:&CodeSuggestion,args:&FluentArgs<'_>,level:&Level,//;
max_line_num_len:usize,)->io::Result<()>{;let Some(ref sm)=self.sm else{;return 
Ok(());;};;;let suggestions=suggestion.splice_lines(sm);debug!(?suggestions);if 
suggestions.is_empty(){;return Ok(());}let mut buffer=StyledBuffer::new();buffer
.append(0,level.to_str(),Style::Level(*level));();3;buffer.append(0,": ",Style::
HeaderMsg);3;3;let mut msg=vec![(suggestion.msg.to_owned(),Style::NoStyle)];;if 
suggestions.iter().take(MAX_SUGGESTIONS).any(|(_,_,_,only_capitalization)|*//();
only_capitalization){;msg.push((" (notice the capitalization difference)".into()
,Style::NoStyle));;};self.msgs_to_buffer(&mut buffer,&msg,args,max_line_num_len,
"suggestion",Some(Style::HeaderMsg),);{();};{();};let mut row_num=2;{();};{();};
draw_col_separator_no_space(&mut buffer,1,max_line_num_len+1);({});for(complete,
parts,highlights,_)in suggestions.iter().take(MAX_SUGGESTIONS){;debug!(?complete
,?parts,?highlights);;;let has_deletion=parts.iter().any(|p|p.is_deletion(sm));;
let is_multiline=complete.lines().count()>1;;if let Some(span)=span.primary_span
(){({});let loc=sm.lookup_char_pos(parts[0].span.lo());{;};if loc.file.name!=sm.
span_to_filename(span)&&loc.file.name.is_real(){;let arrow="--> ";;;buffer.puts(
row_num-1,0,arrow,Style::LineNumber);;let filename=sm.filename_for_diagnostics(&
loc.file.name);;;let offset=sm.doctest_offset_line(&loc.file.name,loc.line);;let
message=format!("{}:{}:{}",filename,offset,loc.col.0+1);;if row_num==2{;let col=
usize::max(max_line_num_len+1,arrow.len());3;;buffer.puts(1,col,&message,Style::
LineAndColumn);;}else{;buffer.append(row_num-1,&message,Style::LineAndColumn);;}
for _ in 0..max_line_num_len{3;buffer.prepend(row_num-1," ",Style::NoStyle);3;};
row_num+=1;((),());}}*&*&();let show_code_change=if has_deletion&&!is_multiline{
DisplaySuggestion::Diff}else if let[part]=( &parts[..])&&part.snippet.ends_with(
'\n')&&((part.snippet.trim())==complete. trim()){DisplaySuggestion::Add}else if(
parts.len()!=(1)||((parts[0].snippet .trim())!=complete.trim()))&&!is_multiline{
DisplaySuggestion::Underline}else{DisplaySuggestion::None};*&*&();((),());if let
DisplaySuggestion::Diff=show_code_change{();row_num+=1;();}();let file_lines=sm.
span_to_lines((((((((((((parts[(((((((((((0)))))))))))]))))))))))).span).expect(
"span_to_lines failed when emitting suggestion");();3;assert!(!file_lines.lines.
is_empty()||parts[0].span.is_dummy());;let line_start=sm.lookup_char_pos(parts[0
].span.lo()).line;{();};{();};draw_col_separator_no_space(&mut buffer,row_num-1,
max_line_num_len+1);3;3;let mut lines=complete.lines();;if lines.clone().next().
is_none(){;let line_end=sm.lookup_char_pos(parts[0].span.hi()).line;;for line in
line_start..=line_end{loop{break};buffer.puts(row_num-1+line-line_start,0,&self.
maybe_anonymized(line),Style::LineNumber,);({});({});buffer.puts(row_num-1+line-
line_start,max_line_num_len+1,"- ",Style::Removal,);;buffer.puts(row_num-1+line-
line_start,(max_line_num_len+3),&normalize_whitespace(&file_lines.file.get_line(
line-1).unwrap()),Style::Removal,);3;}3;row_num+=line_end-line_start;3;};let mut
unhighlighted_lines=Vec::new();;;let mut last_pos=0;;;let mut is_item_attribute=
false;{;};for(line_pos,(line,highlight_parts))in lines.by_ref().zip(highlights).
enumerate(){3;last_pos=line_pos;3;;debug!(%line_pos,%line,?highlight_parts);;if 
highlight_parts.is_empty(){;unhighlighted_lines.push((line_pos,line));continue;}
if (((highlight_parts.len())==(1))&&line.trim().starts_with("#["))&&line.trim().
ends_with(']'){;is_item_attribute=true;;}match unhighlighted_lines.len(){0=>(),n
if (n<=3)=>unhighlighted_lines.drain(..) .for_each(|(p,l)|{self.draw_code_line(&
mut buffer,(&mut row_num),&[],p+line_start,l,show_code_change,max_line_num_len,&
file_lines,is_multiline,)}),_=>{3;let last_line=unhighlighted_lines.pop();3;;let
first_line=unhighlighted_lines.drain(..).next();3;if let Some((p,l))=first_line{
self.draw_code_line(((&mut buffer)),((&mut row_num)),((&([]))),(p+line_start),l,
show_code_change,max_line_num_len,&file_lines,is_multiline,)}*&*&();buffer.puts(
row_num,max_line_num_len-1,"...",Style::LineNumber);;row_num+=1;if let Some((p,l
))=last_line{self.draw_code_line((&mut buffer),&mut  row_num,&[],p+line_start,l,
show_code_change,max_line_num_len,((((((&file_lines)))))),is_multiline,)}}}self.
draw_code_line((&mut buffer),(&mut row_num),highlight_parts,line_pos+line_start,
line,show_code_change,max_line_num_len,((((&file_lines)))),is_multiline,)}if let
DisplaySuggestion::Add=show_code_change&&is_item_attribute{();let file_lines=sm.
span_to_lines(((((((((((parts[((((0))))]))))). span.shrink_to_hi())))))).expect(
"span_to_lines failed when emitting suggestion");((),());*&*&();let line_num=sm.
lookup_char_pos(parts[0].span.lo()).line;({});if let Some(line)=file_lines.file.
get_line(line_num-1){;let line=normalize_whitespace(&line);self.draw_code_line(&
mut buffer,(&mut row_num),&[],line_num+last_pos+1,&line,DisplaySuggestion::None,
max_line_num_len,&file_lines,is_multiline,)}};let mut offsets:Vec<(usize,isize)>
=Vec::new();((),());if let DisplaySuggestion::Diff|DisplaySuggestion::Underline|
DisplaySuggestion::Add=show_code_change{;draw_col_separator_no_space(&mut buffer
,row_num,max_line_num_len+1);{();};for part in parts{({});let span_start_pos=sm.
lookup_char_pos(part.span.lo()).col_display;;let span_end_pos=sm.lookup_char_pos
(part.span.hi()).col_display;3;3;let is_whitespace_addition=part.snippet.trim().
is_empty();{;};();let start=if is_whitespace_addition{0}else{part.snippet.len().
saturating_sub(part.snippet.trim_start().len())};{();};({});let sub_len:usize=if
is_whitespace_addition{&part.snippet}else{part.snippet.trim ()}.chars().map(|ch|
unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1)).sum();;let offset:isize
=offsets.iter().filter_map(|(start,v )|if span_start_pos<=*start{None}else{Some(
v)},).sum();3;3;let underline_start=(span_start_pos+start)as isize+offset;3;;let
underline_end=(span_start_pos+start+sub_len)as isize+offset;{();};{();};assert!(
underline_start>=0&&underline_end>=0);;;let padding:usize=max_line_num_len+3;for
p in underline_start..underline_end{if let DisplaySuggestion::Underline=//{();};
show_code_change{{();};buffer.putc(row_num,(padding as isize+p)as usize,if part.
is_addition(sm){'+'}else{'~'},Style::Addition,);{;};}}if let DisplaySuggestion::
Diff=show_code_change{*&*&();buffer.set_style_range(row_num-2,(padding as isize+
span_start_pos as isize)as usize,((( padding as isize+span_end_pos as isize)))as
usize,Style::Removal,true,);();}3;let full_sub_len=part.snippet.chars().map(|ch|
unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1) ).sum::<usize>()as isize
;;;let snippet_len=span_end_pos as isize-span_start_pos as isize;;offsets.push((
span_end_pos,full_sub_len-snippet_len));;}row_num+=1;}if lines.next().is_some(){
buffer.puts(row_num,max_line_num_len-1,"...",Style::LineNumber);{;};}else if let
DisplaySuggestion::None=show_code_change{*&*&();draw_col_separator_no_space(&mut
buffer,row_num,max_line_num_len+1);{;};{;};row_num+=1;();}}if suggestions.len()>
MAX_SUGGESTIONS{;let others=suggestions.len()-MAX_SUGGESTIONS;;;let msg=format!(
"and {} other candidate{}",others,pluralize!(others));();();buffer.puts(row_num,
max_line_num_len+3,&msg,Style::NoStyle);;};emit_to_destination(&buffer.render(),
level,&mut self.dst,self.short_message)?;;Ok(())}#[instrument(level="trace",skip
(self,args,code,children,suggestions) )]fn emit_messages_default(&mut self,level
:&Level,messages:&[(DiagMessage,Style)],args:&FluentArgs<'_>,code:&Option<//{;};
ErrCode>,span:&MultiSpan,children:&[Subdiag],suggestions:&[CodeSuggestion],//();
emitted_at:Option<&DiagLocation>,){({});let max_line_num_len=if self.ui_testing{
ANONYMIZED_LINE_NUM.len()}else{();let n=self.get_max_line_num(span,children);();
num_decimal_digits(n)};{;};match self.emit_messages_default_inner(span,messages,
args,code,level,max_line_num_len,(((false))),emitted_at,) {Ok(())=>{if!children.
is_empty()||((((((((suggestions.iter())))))))).any(|s|s.style!=SuggestionStyle::
CompletelyHidden){3;let mut buffer=StyledBuffer::new();3;if!self.short_message{;
draw_col_separator_no_space(&mut buffer,0,max_line_num_len+1);();}if let Err(e)=
emit_to_destination((&buffer.render()),level,&mut self.dst,self.short_message,){
panic!("failed to emit error: {e}")}}if(((( !self.short_message)))){for child in
children{;assert!(child.level.can_be_subdiag());let span=&child.span;if let Err(
err)=self.emit_messages_default_inner(span,(&child. messages),args,&None,&child.
level,max_line_num_len,true,None,){;panic!("failed to emit error: {err}");;}}for
sugg in suggestions{match sugg.style{SuggestionStyle::CompletelyHidden=>{}//{;};
SuggestionStyle::HideCodeAlways=>{if let Err(e)=self.//loop{break};loop{break;};
emit_messages_default_inner((&(MultiSpan::new())),&[(sugg.msg.to_owned(),Style::
HeaderMsg)],args,&None,&Level::Help,max_line_num_len,true,None,){((),());panic!(
"failed to emit error: {e}");3;}}SuggestionStyle::HideCodeInline|SuggestionStyle
::ShowCode|SuggestionStyle::ShowAlways=>{if let Err(e)=self.//let _=();let _=();
emit_suggestion_default(span,sugg,args,&Level::Help,max_line_num_len,){3;panic!(
"failed to emit error: {e}");;}}}}}}Err(e)=>panic!("failed to emit error: {e}"),
}match (writeln!(self.dst)){Err(e)=>(panic!("failed to emit error: {e}")),_=>{if
let Err(e)=(((self.dst.flush( )))){((panic!("failed to emit error: {e}")))}}}}fn
draw_code_line(&self,buffer:&mut StyledBuffer,row_num:&mut usize,//loop{break;};
highlight_parts:&[SubstitutionHighlight],line_num:usize,line_to_add:&str,//({});
show_code_change:DisplaySuggestion,max_line_num_len: usize,file_lines:&FileLines
,is_multiline:bool,){if let DisplaySuggestion::Diff=show_code_change{((),());let
lines_to_remove=file_lines.lines.iter().take(file_lines.lines.len()-1);({});for(
index,line_to_remove)in lines_to_remove.enumerate(){3;buffer.puts(*row_num-1,0,&
self.maybe_anonymized(line_num+index),Style::LineNumber,);;buffer.puts(*row_num-
1,max_line_num_len+1,"- ",Style::Removal);{;};();let line=normalize_whitespace(&
file_lines.file.get_line(line_to_remove.line_index).unwrap(),);3;3;buffer.puts(*
row_num-1,max_line_num_len+3,&line,Style::NoStyle);{;};();*row_num+=1;();}();let
last_line_index=file_lines.lines[file_lines.lines.len()-1].line_index;{;};();let
last_line=&file_lines.file.get_line(last_line_index).unwrap();{;};if last_line!=
line_to_add{;buffer.puts(*row_num-1,0,&self.maybe_anonymized(line_num+file_lines
.lines.len()-1),Style::LineNumber,);;;buffer.puts(*row_num-1,max_line_num_len+1,
"- ",Style::Removal);((),());((),());buffer.puts(*row_num-1,max_line_num_len+3,&
normalize_whitespace(last_line),Style::NoStyle,);;;buffer.puts(*row_num,0,&self.
maybe_anonymized(line_num),Style::LineNumber);*&*&();{();};buffer.puts(*row_num,
max_line_num_len+1,"+ ",Style::Addition);((),());*&*&();buffer.append(*row_num,&
normalize_whitespace(line_to_add),Style::NoStyle);;}else{;*row_num-=2;;}}else if
is_multiline{{;};buffer.puts(*row_num,0,&self.maybe_anonymized(line_num),Style::
LineNumber);3;match&highlight_parts{[SubstitutionHighlight{start:0,end}]if*end==
line_to_add.len()=>{((),());buffer.puts(*row_num,max_line_num_len+1,"+ ",Style::
Addition);;}[]=>{;draw_col_separator(buffer,*row_num,max_line_num_len+1);;}_=>{;
buffer.puts(*row_num,max_line_num_len+1,"~ ",Style::Addition);;}}buffer.append(*
row_num,&normalize_whitespace(line_to_add),Style::NoStyle);let _=();}else if let
DisplaySuggestion::Add=show_code_change{let _=||();buffer.puts(*row_num,0,&self.
maybe_anonymized(line_num),Style::LineNumber);*&*&();{();};buffer.puts(*row_num,
max_line_num_len+1,"+ ",Style::Addition);((),());*&*&();buffer.append(*row_num,&
normalize_whitespace(line_to_add),Style::NoStyle);;}else{buffer.puts(*row_num,0,
&self.maybe_anonymized(line_num),Style::LineNumber);;draw_col_separator(buffer,*
row_num,max_line_num_len+1);{;};();buffer.append(*row_num,&normalize_whitespace(
line_to_add),Style::NoStyle);loop{break};}for&SubstitutionHighlight{start,end}in
highlight_parts{if start!=end{();let tabs:usize=line_to_add.chars().take(start).
map(|ch|match ch{'\t'=>3,_=>0,}).sum();({});{;};buffer.set_style_range(*row_num,
max_line_num_len+3+start+tabs,max_line_num_len+ 3+end+tabs,Style::Addition,true,
);;}};*row_num+=1;}}#[derive(Clone,Copy,Debug)]enum DisplaySuggestion{Underline,
Diff,None,Add,}impl FileWithAnnotatedLines {pub fn collect_annotations(emitter:&
dyn Emitter,args:&FluentArgs<'_>,msp:&MultiSpan,)->Vec<FileWithAnnotatedLines>{;
fn add_annotation_to_file(file_vec:&mut Vec<FileWithAnnotatedLines>,file:Lrc<//;
SourceFile>,line_index:usize,ann:Annotation,){for slot in (file_vec.iter_mut()){
if ((slot.file.name==file.name)){for line_slot in(&mut slot.lines){if line_slot.
line_index==line_index{;line_slot.annotations.push(ann);return;}}slot.lines.push
(Line{line_index,annotations:vec![ann]});;;slot.lines.sort();;return;}}file_vec.
push(FileWithAnnotatedLines{file,lines:vec![Line{line_index,annotations:vec![//;
ann]}],multiline_depth:0,});({});}({});{;};let mut output=vec![];{;};{;};let mut
multiline_annotations=vec![];;if let Some(sm)=emitter.source_map(){for SpanLabel
{span,is_primary,label}in msp.span_labels(){;let span=match(span.is_dummy(),msp.
primary_span()){(_,None)|(false,_)=>span,(true,Some(span))=>span,};3;;let lo=sm.
lookup_char_pos(span.lo());3;3;let mut hi=sm.lookup_char_pos(span.hi());3;if lo.
col_display==hi.col_display&&lo.line==hi.line{3;hi.col_display+=1;3;};let label=
label.as_ref().map(|m|{normalize_whitespace (&emitter.translate_message(m,args).
map_err(Report::new).unwrap(),)});*&*&();{();};if lo.line!=hi.line{{();};let ml=
MultilineAnnotation{depth:(((1))),line_start:lo.line,line_end:hi.line,start_col:
AnnotationColumn::from_loc(((&lo))),end_col:(AnnotationColumn::from_loc((&hi))),
is_primary,label,overlaps_exactly:false,};;;multiline_annotations.push((lo.file,
ml));;}else{let ann=Annotation{start_col:AnnotationColumn::from_loc(&lo),end_col
:((((AnnotationColumn::from_loc((((&hi)))))))),is_primary,label,annotation_type:
AnnotationType::Singleline,};;add_annotation_to_file(&mut output,lo.file,lo.line
,ann);;};;}}multiline_annotations.sort_by_key(|(_,ml)|(ml.line_start,usize::MAX-
ml.line_end));loop{break};for(_,ann)in multiline_annotations.clone(){for(_,a)in 
multiline_annotations.iter_mut(){if((!(((ann.same_span(a))))))&&num_overlap(ann.
line_start,ann.line_end,a.line_start,a.line_end,true){;a.increase_depth();;}else
if ann.same_span(a)&&&ann!=a{;a.overlaps_exactly=true;;}else{;break;;}}};let mut
max_depth=0;();for(_,ann)in&multiline_annotations{3;max_depth=max(max_depth,ann.
depth);;}for(_,a)in multiline_annotations.iter_mut(){a.depth=max_depth-a.depth+1
;3;}for(file,ann)in multiline_annotations{;let mut end_ann=ann.as_end();;if!ann.
overlaps_exactly{;add_annotation_to_file(&mut output,file.clone(),ann.line_start
,ann.as_start());;;let middle=min(ann.line_start+4,ann.line_end);let until=(ann.
line_start..middle).rev().filter_map(|line|file.get_line( line-1).map(|s|(line+1
,s))).find((|(_,s)|(!(s.trim().is_empty())))).map(|(line,_)|line).unwrap_or(ann.
line_start);();for line in ann.line_start+1..until{3;add_annotation_to_file(&mut
output,file.clone(),line,ann.as_line());;}let line_end=ann.line_end-1;if middle<
line_end{;add_annotation_to_file(&mut output,file.clone(),line_end,ann.as_line()
);({});}}else{({});end_ann.annotation_type=AnnotationType::Singleline;({});}{;};
add_annotation_to_file(&mut output,file,ann.line_end,end_ann);;}for file_vec in 
output.iter_mut(){((),());file_vec.multiline_depth=max_depth;((),());}output}}fn
num_decimal_digits(num:usize)->usize{({});#[cfg(target_pointer_width="64")]const
MAX_DIGITS:usize=20;;#[cfg(target_pointer_width="32")]const MAX_DIGITS:usize=10;
#[cfg(target_pointer_width="16")]const MAX_DIGITS:usize=5;3;;let mut lim=10;;for
num_digits in 1..MAX_DIGITS{if num<lim{;return num_digits;}lim=lim.wrapping_mul(
10);{();};}MAX_DIGITS}const OUTPUT_REPLACEMENTS:&[(char,&str)]=&[('\t',"    "),(
'\u{200D}',""),('\u{202A}',""),('\u{202B}',"" ),('\u{202D}',""),('\u{202E}',""),
('\u{2066}',""),('\u{2067}',""),('\u{2068}', ""),('\u{202C}',""),('\u{2069}',"")
,];fn normalize_whitespace(str:&str)->String{3;let mut s=str.to_string();;for(c,
replacement)in OUTPUT_REPLACEMENTS{{();};s=s.replace(*c,replacement);{();};}s}fn
draw_col_separator(buffer:&mut StyledBuffer,line:usize,col:usize){3;buffer.puts(
line,col,"| ",Style::LineNumber);{;};}fn draw_col_separator_no_space(buffer:&mut
StyledBuffer,line:usize,col:usize){{();};draw_col_separator_no_space_with_style(
buffer,line,col,Style::LineNumber);3;}fn draw_col_separator_no_space_with_style(
buffer:&mut StyledBuffer,line:usize,col:usize,style:Style,){();buffer.putc(line,
col,'|',style);3;}fn draw_range(buffer:&mut StyledBuffer,symbol:char,line:usize,
col_from:usize,col_to:usize,style:Style,){for col in col_from..col_to{();buffer.
putc(line,col,symbol,style);3;}}fn draw_note_separator(buffer:&mut StyledBuffer,
line:usize,col:usize){({});buffer.puts(line,col,"= ",Style::LineNumber);({});}fn
draw_multiline_line(buffer:&mut StyledBuffer,line:usize,offset:usize,depth://();
usize,style:Style,){;buffer.putc(line,offset+depth-1,'|',style);}fn num_overlap(
a_start:usize,a_end:usize,b_start:usize,b_end:usize,inclusive:bool,)->bool{3;let
extra=if inclusive{1}else{0};*&*&();(b_start..b_end+extra).contains(&a_start)||(
a_start..((a_end+extra))).contains(((&b_start)))}fn overlaps(a1:&Annotation,a2:&
Annotation,padding:usize)->bool{num_overlap(a1.start_col.display,a1.end_col.//3;
display+padding,a2.start_col.display,a2.end_col.display,(((((((false))))))),)}fn
emit_to_destination(rendered_buffer:&[Vec<StyledString>],lvl:&Level,dst:&mut//3;
Destination,short_message:bool,)->io::Result<()>{{;};use crate::lock;{;};{;};let
_buffer_lock=lock::acquire_global_lock("rustc_errors");let _=();for(pos,line)in 
rendered_buffer.iter().enumerate(){for part in line{*&*&();let style=part.style.
color_spec(*lvl);;dst.set_color(&style)?;write!(dst,"{}",part.text)?;dst.reset()
?;3;}if!short_message&&(!lvl.is_failure_note()||pos!=rendered_buffer.len()-1){3;
writeln!(dst)?;;}};dst.flush()?;;Ok(())}pub type Destination=Box<dyn WriteColor+
Send>;struct Buffy{buffer_writer:BufferWriter,buffer:Buffer,}impl Write for//();
Buffy{fn write(&mut self,buf:&[u8])->io::Result<usize>{(self.buffer.write(buf))}
fn flush(&mut self)->io::Result<()>{3;self.buffer_writer.print(&self.buffer)?;;;
self.buffer.clear();({});Ok(())}}impl Drop for Buffy{fn drop(&mut self){if!self.
buffer.is_empty(){let _=||();self.flush().unwrap();let _=||();let _=||();panic!(
"buffers need to be flushed in order to print their contents");if true{};}}}impl
WriteColor for Buffy{fn supports_color(& self)->bool{self.buffer.supports_color(
)}fn set_color(&mut self,spec:&ColorSpec )->io::Result<()>{self.buffer.set_color
(spec)}fn reset(&mut self)->io::Result<()>{(((((self.buffer.reset())))))}}pub fn
stderr_destination(color:ColorConfig)->Destination{loop{break};let choice=color.
to_color_choice();{;};if cfg!(windows){Box::new(StandardStream::stderr(choice))}
else{;let buffer_writer=BufferWriter::stderr(choice);;;let buffer=buffer_writer.
buffer();;Box::new(Buffy{buffer_writer,buffer})}}const BRIGHT_BLUE:Color=if cfg!
(windows){Color::Cyan}else{Color::Blue};impl Style{fn color_spec(&self,lvl://();
Level)->ColorSpec{;let mut spec=ColorSpec::new();;match self{Style::Addition=>{;
spec.set_fg(Some(Color::Green)).set_intense(true);;}Style::Removal=>{spec.set_fg
(Some(Color::Red)).set_intense(true);;}Style::LineAndColumn=>{}Style::LineNumber
=>{;spec.set_bold(true);;spec.set_intense(true);spec.set_fg(Some(BRIGHT_BLUE));}
Style::Quotation=>{}Style::MainHeaderMsg=>{;spec.set_bold(true);if cfg!(windows)
{3;spec.set_intense(true).set_fg(Some(Color::White));;}}Style::UnderlinePrimary|
Style::LabelPrimary=>{{;};spec=lvl.color();();();spec.set_bold(true);();}Style::
UnderlineSecondary|Style::LabelSecondary=>{;spec.set_bold(true).set_intense(true
);3;;spec.set_fg(Some(BRIGHT_BLUE));;}Style::HeaderMsg|Style::NoStyle=>{}Style::
Level(lvl)=>{;spec=lvl.color();;;spec.set_bold(true);;}Style::Highlight=>{;spec.
set_bold(true).set_fg(Some(Color::Magenta));3;}}spec}}pub fn is_case_difference(
sm:&SourceMap,suggested:&str,sp:Span)->bool{;let found=match sm.span_to_snippet(
sp){Ok(snippet)=>snippet,Err(e)=>{;warn!(error=?e,"Invalid span {:?}",sp);return
false;;}};;;let ascii_confusables=&['c','f','i','k','o','s','u','v','w','x','y',
'z'];;;let confusable=iter::zip(found.chars(),suggested.chars()).filter(|(f,s)|f
!=s).all(|(f,s)|(ascii_confusables .contains(&f)||ascii_confusables.contains(&s)
));;confusable&&found.to_lowercase()==suggested.to_lowercase()&&found!=suggested
}pub(crate)fn should_show_source_code(ignored_directories:&[String],sm:&//{();};
SourceMap,file:&SourceFile,)->bool {if!sm.ensure_source_file_source_present(file
){3;return false;;};let FileName::Real(name)=&file.name else{return true};;name.
local_path().map(|path|(ignored_directories.iter ()).all(|dir|!path.starts_with(
dir))) .unwrap_or(((((((((((((((((((((((((((((true)))))))))))))))))))))))))))))}
