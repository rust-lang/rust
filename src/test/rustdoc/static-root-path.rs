// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matchesraw - '"/cache/main\.js"'
// @!matches - '"\.\./main\.js"'
// @matchesraw - 'data-root-path="\.\./"'
// @!matches - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matchesraw - '"/cache/source-script\.js"'
// @!matches - '"\.\./\.\./source-script\.js"'
// @matchesraw - '"\.\./\.\./source-files.js"'
// @!matches - '"/cache/source-files\.js"'

// @has settings.html
// @matchesraw - '/cache/settings\.js'
// @!matches - '\./settings\.js'
