//@ is "$.index[?(@.name=='not')].attrs" []
//@ is "$.index[?(@.name=='not')].deprecation" null
pub fn not() {}

//@ is "$.index[?(@.name=='raw')].attrs" []
//@ is "$.index[?(@.name=='raw')].deprecation" '{"since": null, "note": null}'
#[deprecated]
pub fn raw() {}

//@ is "$.index[?(@.name=='equals_string')].attrs" []
//@ is "$.index[?(@.name=='equals_string')].deprecation" '{"since": null, "note": "here is a reason"}'
#[deprecated = "here is a reason"]
pub fn equals_string() {}

//@ is "$.index[?(@.name=='since')].attrs" []
//@ is "$.index[?(@.name=='since')].deprecation" '{"since": "yoinks ago", "note": null}'
#[deprecated(since = "yoinks ago")]
pub fn since() {}

//@ is "$.index[?(@.name=='note')].attrs" []
//@ is "$.index[?(@.name=='note')].deprecation" '{"since": null, "note": "7"}'
#[deprecated(note = "7")]
pub fn note() {}

//@ is "$.index[?(@.name=='since_and_note')].attrs" []
//@ is "$.index[?(@.name=='since_and_note')].deprecation" '{"since": "tomorrow", "note": "sorry"}'
#[deprecated(since = "tomorrow", note = "sorry")]
pub fn since_and_note() {}

//@ is "$.index[?(@.name=='note_and_since')].attrs" []
//@ is "$.index[?(@.name=='note_and_since')].deprecation" '{"since": "a year from tomorrow", "note": "your welcome"}'
#[deprecated(note = "your welcome", since = "a year from tomorrow")]
pub fn note_and_since() {}

//@ is "$.index[?(@.name=='neither_but_parens')].attrs" []
//@ is "$.index[?(@.name=='neither_but_parens')].deprecation" '{"since": null, "note": null}'
#[deprecated()]
pub fn neither_but_parens() {}
