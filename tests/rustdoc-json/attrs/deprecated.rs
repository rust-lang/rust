//@ eq .index[] | select(.name == "not") | [.attrs, .deprecation], [[], null]
pub fn not() {}

//@ eq .index[] | select(.name == "raw") | [.attrs, .deprecation], [[], {"since": null, "note": null}]
#[deprecated]
pub fn raw() {}

//@ eq .index[] | select(.name == "equals_string") | [.attrs, .deprecation], [[], {"since": null, "note": "here is a reason"}]
#[deprecated = "here is a reason"]
pub fn equals_string() {}

//@ eq .index[] | select(.name == "since") | [.attrs, .deprecation], [[], {"since": "yoinks ago", "note": null}]
#[deprecated(since = "yoinks ago")]
pub fn since() {}

//@ eq .index[] | select(.name == "note") | [.attrs, .deprecation], [[], {"since": null, "note": "7"}]
#[deprecated(note = "7")]
pub fn note() {}

//@ eq .index[] | select(.name == "since_and_note") | [.attrs, .deprecation], [[], {"since": "tomorrow", "note": "sorry"}]
#[deprecated(since = "tomorrow", note = "sorry")]
pub fn since_and_note() {}

//@ eq .index[] | select(.name == "note_and_since") | [.attrs, .deprecation], [[], {"since": "a year from tomorrow", "note": "your welcome"}]
#[deprecated(note = "your welcome", since = "a year from tomorrow")]
pub fn note_and_since() {}

//@ eq .index[] | select(.name == "neither_but_parens") | [.attrs, .deprecation], [[], {"since": null, "note": null}]
#[deprecated()]
pub fn neither_but_parens() {}
