// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matchesraw - '"/cache/main-'
// @!matchesraw - '"\.\./main'
// @matchesraw - 'data-root-path="\.\./"'
// @!matchesraw - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matchesraw - '"/cache/src-script-'
// @!matchesraw - '"\.\./\.\./src-script'
// @matchesraw - '"\.\./\.\./src-files.js"'
// @!matchesraw - '"/cache/src-files\.js"'

// @has settings.html
// @matchesraw - '/cache/settings-'
// @!matchesraw - '\../settings'
