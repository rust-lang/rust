//@ is "$.index[*][?(@.name=='not')].attrs" '[]'
//@ is "$.index[*][?(@.name=='not')].deprecation" null
pub fn not() {}

//@ is "$.index[*][?(@.name=='raw')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]\n"]'
//@ is "$.index[*][?(@.name=='raw')].deprecation" '{"since": null, "note": null}'
#[deprecated]
pub fn raw() {}

//@ is "$.index[*][?(@.name=='equals_string')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since: Unspecified, note:\n\"here is a reason\"}}]\n"]'
//@ is "$.index[*][?(@.name=='equals_string')].deprecation" '{"since": null, "note": "here is a reason"}'
#[deprecated = "here is a reason"]
pub fn equals_string() {}

//@ is "$.index[*][?(@.name=='since')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since:\nNonStandard(\"yoinks ago\")}}]\n"]'
//@ is "$.index[*][?(@.name=='since')].deprecation" '{"since": "yoinks ago", "note": null}'
#[deprecated(since = "yoinks ago")]
pub fn since() {}

//@ is "$.index[*][?(@.name=='note')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since: Unspecified, note:\n\"7\"}}]\n"]'
//@ is "$.index[*][?(@.name=='note')].deprecation" '{"since": null, "note": "7"}'
#[deprecated(note = "7")]
pub fn note() {}

//@ is "$.index[*][?(@.name=='since_and_note')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since:\nNonStandard(\"tomorrow\"), note: \"sorry\"}}]\n"]'
//@ is "$.index[*][?(@.name=='since_and_note')].deprecation" '{"since": "tomorrow", "note": "sorry"}'
#[deprecated(since = "tomorrow", note = "sorry")]
pub fn since_and_note() {}

//@ is "$.index[*][?(@.name=='note_and_since')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since:\nNonStandard(\"a year from tomorrow\"), note: \"your welcome\"}}]\n"]'
//@ is "$.index[*][?(@.name=='note_and_since')].deprecation" '{"since": "a year from tomorrow", "note": "your welcome"}'
#[deprecated(note = "your welcome", since = "a year from tomorrow")]
pub fn note_and_since() {}

//@ is "$.index[*][?(@.name=='neither_but_parens')].attrs" '["#[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]\n"]'
//@ is "$.index[*][?(@.name=='neither_but_parens')].deprecation" '{"since": null, "note": null}'
#[deprecated()]
pub fn neither_but_parens() {}
