./src/test/run-pass/syntax-extension-shell.rs:6:18:6:19: [1;31merror:[0m expecting (, found {
./src/test/run-pass/syntax-extension-shell.rs:6   auto s = #shell { uname -a };
                                                                  ^
rt: ---
rt: f00e:main:main:                   upcall fail 'explicit failure', src/comp/syntax/parse/parser.rs:112
rt: f00e:main:                        domain main @0x910c04c root task failed
