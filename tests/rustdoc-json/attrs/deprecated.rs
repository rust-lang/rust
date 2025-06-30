//@ arg not .index[] | select(.name == "not")
//@ jq $not.attrs == []
//@ jq $not.deprecation == null
pub fn not() {}

//@ arg raw .index[] | select(.name == "raw")
//@ jq $raw.attrs == []
//@ jq $raw.deprecation == {"since": null, "note": null}
#[deprecated]
pub fn raw() {}

//@ arg equals_string .index[] | select(.name == "equals_string")
//@ jq $equals_string.attrs == []
//@ jq $equals_string.deprecation == {"since": null, "note": "here is a reason"}
#[deprecated = "here is a reason"]
pub fn equals_string() {}

//@ arg since .index[] | select(.name == "since")
//@ jq $since.attrs == []
//@ jq $since.deprecation == {"since": "yoinks ago", "note": null}
#[deprecated(since = "yoinks ago")]
pub fn since() {}

//@ arg note .index[] | select(.name == "note")
//@ jq $note.attrs == []
//@ jq $note.deprecation == {"since": null, "note": "7"}
#[deprecated(note = "7")]
pub fn note() {}

//@ arg since_and_note .index[] | select(.name == "since_and_note")
//@ jq $since_and_note.attrs == []
//@ jq $since_and_note.deprecation == {"since": "tomorrow", "note": "sorry"}
#[deprecated(since = "tomorrow", note = "sorry")]
pub fn since_and_note() {}

//@ arg note_and_since .index[] | select(.name == "note_and_since")
//@ jq $note_and_since.attrs == []
//@ jq $note_and_since.deprecation == {"since": "a year from tomorrow", "note": "your welcome"}
#[deprecated(note = "your welcome", since = "a year from tomorrow")]
pub fn note_and_since() {}

//@ arg neither_but_parens .index[] | select(.name == "neither_but_parens")
//@ jq $neither_but_parens.attrs == []
//@ jq $neither_but_parens.deprecation == {"since": null, "note": null}
#[deprecated()]
pub fn neither_but_parens() {}
