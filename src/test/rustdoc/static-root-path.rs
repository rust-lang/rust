// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matchesraw - '"/cache/main\.js"'
// @!matchesraw - '"\.\./main\.js"'
// @matchesraw - 'data-root-path="\.\./"'
// @!matchesraw - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matchesraw - '"/cache/source-script\.js"'
// @!matchesraw - '"\.\./\.\./source-script\.js"'
// @matchesraw - '"\.\./\.\./source-files.js"'
// @!matchesraw - '"/cache/source-files\.js"'

// @has settings.html
// @matchesraw - '/cache/settings\.js'
// @!matchesraw - '\./settings\.js'
