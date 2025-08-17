use std::cell::RefCell;

use super::*;

#[test]
fn test_extract_tags_nested_unclosed() {
    let mut tagp = TagParser::new();
    let diags = RefCell::new(Vec::new());
    let dox = "<div>\n<br</div>";
    tagp.extract_tags(dox, 0..dox.len(), &mut None, &|s, r, b| {
        diags.borrow_mut().push((s, r.clone(), b));
    });
    assert_eq!(diags.borrow().len(), 1, "did not get expected diagnostics: {diags:?}");
    assert_eq!(diags.borrow()[0].1, 6..9)
}

#[test]
fn test_extract_tags_taglike_in_attr() {
    let mut tagp = TagParser::new();
    let diags = RefCell::new(Vec::new());
    let dox = "<img src='<div>'>";
    tagp.extract_tags(dox, 0..dox.len(), &mut None, &|s, r, b| {
        diags.borrow_mut().push((s, r.clone(), b));
    });
    assert_eq!(diags.borrow().len(), 0, "unexpected diagnostics: {diags:?}");
}

#[test]
fn test_extract_tags_taglike_in_multiline_attr() {
    let mut tagp = TagParser::new();
    let diags = RefCell::new(Vec::new());
    let dox = "<img src=\"\nasd\n<div>\n\">";
    tagp.extract_tags(dox, 0..dox.len(), &mut None, &|s, r, b| {
        diags.borrow_mut().push((s, r.clone(), b));
    });
    assert_eq!(diags.borrow().len(), 0, "unexpected diagnostics: {diags:?}");
}

#[test]
fn test_extract_tags_taglike_in_multievent_attr() {
    let mut tagp = TagParser::new();
    let diags = RefCell::new(Vec::new());
    let dox = "<img src='<div>'>";
    let split_point = 10;
    let mut p = |range: Range<usize>| {
        tagp.extract_tags(&dox[range.clone()], range, &mut None, &|s, r, b| {
            diags.borrow_mut().push((s, r.clone(), b));
        })
    };
    p(0..split_point);
    p(split_point..dox.len());
    assert_eq!(diags.borrow().len(), 0, "unexpected diagnostics: {diags:?}");
}

#[test]
fn test_extract_tags_taglike_in_multiline_multievent_attr() {
    let mut tagp = TagParser::new();
    let diags = RefCell::new(Vec::new());
    let dox = "<img src='\n foo:\n </div>\n <p/>\n <div>\n'>";
    let mut p = |range: Range<usize>| {
        tagp.extract_tags(&dox[range.clone()], range, &mut None, &|s, r, b| {
            diags.borrow_mut().push((s, r.clone(), b));
        })
    };
    let mut offset = 0;
    for ln in dox.split_inclusive('\n') {
        let new_offset = offset + ln.len();
        p(offset..new_offset);
        offset = new_offset;
    }
    assert_eq!(diags.borrow().len(), 0, "unexpected diagnostics: {diags:?}");
    assert_eq!(tagp.tags.len(), 1);
}
