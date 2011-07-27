./src/test/run-pass/task-killjoin.rs:22:36:22:46: [1;31merror:[0m expecting (, found supervised
./src/test/run-pass/task-killjoin.rs:22     let task t = spawn "supervised" supervised();
                                                                            ^~~~~~~~~~
rt: ---
rt: f00e:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: f00e:main:                        domain main @0xa7b604c root task failed
