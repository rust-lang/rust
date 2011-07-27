./src/test/compile-fail/attr-dangling-in-mod.rs:7:14:7:14: [1;31merror:[0m expected item but found <eof>
./src/test/compile-fail/attr-dangling-in-mod.rs:7 #[foo = "bar"]
                                                                ^
rt: ---
rt: 0bb1:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: 0bb1:main:                        domain main @0xa2d504c root task failed
