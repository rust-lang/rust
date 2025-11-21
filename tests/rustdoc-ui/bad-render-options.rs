// regression test for https://github.com/rust-lang/rust/issues/149187

#![doc(html_favicon_url)] //~ ERROR: `doc(html_favicon_url)` expects a string value [invalid_doc_attributes]
#![doc(html_logo_url)] //~ ERROR: `doc(html_logo_url)` expects a string value [invalid_doc_attributes]
#![doc(html_playground_url)] //~ ERROR: `doc(html_playground_url)` expects a string value [invalid_doc_attributes]
#![doc(issue_tracker_base_url)] //~ ERROR expects a string value
#![doc(html_favicon_url = 1)] //~ ERROR expects a string value
#![doc(html_logo_url = 2)] //~ ERROR expects a string value
#![doc(html_playground_url = 3)] //~ ERROR expects a string value
#![doc(issue_tracker_base_url = 4)] //~ ERROR expects a string value
#![doc(html_no_source = "asdf")] //~ ERROR `doc(html_no_source)` does not accept a value [invalid_doc_attributes]
