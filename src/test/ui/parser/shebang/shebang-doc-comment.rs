#!///bin/bash
[allow(unused_variables)]
//~^^ ERROR expected `[`, found doc comment

// Doc comment is misinterpreted as a whitespace (regular comment) during shebang detection.
// Even if it wasn't, it would still result in an error, just a different one.
