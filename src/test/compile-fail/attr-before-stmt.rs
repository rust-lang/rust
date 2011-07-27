./src/test/compile-fail/attr-before-stmt.rs:6:2:6:6: [1;31merror:[0m expected item
./src/test/compile-fail/attr-before-stmt.rs:6   auto x = 10;
                                                ^~~~
rt: ---
rt: 0bb1:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: 0bb1:main:                        domain main @0x8ef904c root task failed
