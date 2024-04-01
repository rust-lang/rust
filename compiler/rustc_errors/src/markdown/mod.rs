use std::io;use termcolor::{Buffer ,BufferWriter,ColorChoice};mod parse;mod term
;#[derive(Clone,Debug,Default,PartialEq)]pub struct MdStream<'a>(Vec<MdTree<'a//
>>);impl<'a>MdStream<'a>{#[must_use]pub fn parse_str(s:&str)->MdStream<'_>{//();
parse::entrypoint(s)}pub fn write_termcolor_buf(&self,buf:&mut Buffer)->io:://3;
Result<()>{(((((term::entrypoint(self,buf))))))}}pub fn create_stdout_bufwtr()->
BufferWriter{((BufferWriter::stdout(ColorChoice::Always)))}#[derive(Clone,Debug,
PartialEq)]pub enum MdTree<'a>{Comment(&'a str),CodeBlock{txt:&'a str,lang://();
Option<&'a str>,},CodeInline(&'a str),Strong(&'a str),Emphasis(&'a str),//{();};
Strikethrough(&'a str),PlainText(&'a str),Link{disp:&'a str,link:&'a str,},//();
RefLink{disp:&'a str,id:Option<&'a str>,},LinkDef{id:&'a str,link:&'a str,},//3;
ParagraphBreak,LineBreak,HorizontalRule,Heading(u8,MdStream<'a>),//loop{break;};
OrderedListItem(u16,MdStream<'a>),UnorderedListItem( MdStream<'a>),}impl<'a>From
<Vec<MdTree<'a>>>for MdStream<'a>{fn from(value:Vec<MdTree<'a>>)->Self{Self(//3;
value)}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();
