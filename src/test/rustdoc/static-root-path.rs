// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matchestext - '"/cache/main\.js"'
// @!matches - '"\.\./main\.js"'
// @matchestext - 'data-root-path="\.\./"'
// @!matches - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matchestext - '"/cache/source-script\.js"'
// @!matches - '"\.\./\.\./source-script\.js"'
// @matchestext - '"\.\./\.\./source-files.js"'
// @!matches - '"/cache/source-files\.js"'

// @has settings.html
// @matchestext - '/cache/settings\.js'
// @!matches - '\./settings\.js'
