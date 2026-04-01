#![feature(custom_inner_attributes)]
#![rustfmt::skip]

#![warn(clippy::doc_comment_double_space_linebreaks)]
#![allow(unused, clippy::empty_docs)]

//~v doc_comment_double_space_linebreaks
//! Should warn on double space linebreaks  
//! in file/module doc comment

/// Should not warn on single-line doc comments
fn single_line() {}

/// Should not warn on single-line doc comments
/// split across multiple lines
fn single_line_split() {}

// Should not warn on normal comments

// note: cargo fmt can remove double spaces from normal and block comments
// Should not warn on normal comments  
// with double spaces at the end of a line  

#[doc = "This is a doc attribute, which should not be linted"]
fn normal_comment() {
   /*
   Should not warn on block comments
   */
  
  /*
  Should not warn on block comments  
  with double space at the end of a line
  */
}

//~v doc_comment_double_space_linebreaks
/// Should warn when doc comment uses double space  
/// as a line-break, even when there are multiple  
/// in a row
fn double_space_doc_comment() {}

/// Should not warn when back-slash is used \
/// as a line-break
fn back_slash_doc_comment() {}

//~v doc_comment_double_space_linebreaks
/// 🌹 are 🟥  
/// 🌷 are 🟦  
/// 📎 is 😎  
/// and so are 🫵  
/// (hopefully no formatting weirdness linting this)
fn multi_byte_chars_tada() {}

macro_rules! macro_that_makes_function {
   () => {
      /// Shouldn't lint on this!  
      /// (hopefully)
      fn my_macro_created_function() {}
   }
}

macro_that_makes_function!();

// dont lint when its alone on a line
///  
fn alone() {}

/// | First column | Second column |  
/// | ------------ | ------------- |  
/// | Not a line   | break when    |  
/// | after a line | in a table    |  
fn table() {}

/// ```text  
/// It's also not a hard line break if  
/// there's two spaces at the end of a  
/// line in a block code.  
/// ```  
fn codeblock() {}

/// It's also not a hard line break `if  
/// there's` two spaces in the middle of inline code.  
fn inline() {}

/// It's also not a hard line break [when](  
/// https://example.com) in a URL.  
fn url() {}

//~v doc_comment_double_space_linebreaks
/// here we mix  
/// double spaces\
/// and also  
/// adding backslash\
/// to some of them  
/// to see how that looks
fn mixed() {}

fn main() {}
