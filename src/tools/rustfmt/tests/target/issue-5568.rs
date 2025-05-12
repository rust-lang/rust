// rustfmt-max_width: 119
// rustfmt-format_code_in_doc_comments: true

mod libs {
    fn mrbgems_sources() {
        [
            "mrbgems/mruby-compiler/core/codegen.c", // Ruby parser and bytecode generation
            "mrbgems/mruby-compiler/core/y.tab.c",   // Ruby parser and bytecode generation
            "mrbgems/mruby-metaprog/src/metaprog.c", // APIs on Kernel and Module for accessing classes and variables
            "mrbgems/mruby-method/src/method.c",     // `Method`, `UnboundMethod`, and method APIs on Kernel and Module
            "mrbgems/mruby-pack/src/pack.c",         // Array#pack and String#unpack
        ]
    }
}
