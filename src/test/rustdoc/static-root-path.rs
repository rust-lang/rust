// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matchesraw - '"/cache/main-'
// @!matchesraw - '"\.\./main'
// @matchesraw - 'data-root-path="\.\./"'
// @!matchesraw - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matchesraw - '"/cache/source-script-'
// @!matchesraw - '"\.\./\.\./source-script'
// @matchesraw - '"\.\./\.\./source-files.js"'
// @!matchesraw - '"/cache/source-files\.js"'

// @has settings.html
// @matchesraw - '/cache/settings-'
// @!matchesraw - '\../settings'
