// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matches - '"/cache/main\.js"'
// @!matches - '"\.\./main\.js"'
// @matches - 'data-root-path="\.\./"'
// @!matches - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matches - '"/cache/source-script\.js"'
// @!matches - '"\.\./\.\./source-script\.js"'
// @matches - '"\.\./\.\./source-files.js"'
// @!matches - '"/cache/source-files\.js"'

// @has settings.html
// @matches - '/cache/settings\.js'
// @!matches - '\./settings\.js'
