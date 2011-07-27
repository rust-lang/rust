./src/test/compile-fail/ext-after-attrib.rs:7:1:7:4: [1;31merror:[0m expecting [, found fmt
./src/test/compile-fail/ext-after-attrib.rs:7 #fmt("baz")
                                               ^~~
rt: ---
rt: f00e:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: f00e:main:                        domain main @0xa47a04c root task failed
