import fs from 'fs';

const dec = new TextDecoder("utf-8");

if (process.argv.length != 3) {
    console.log("Usage: node verify.mjs <wasm-file>");
    process.exit(0);
}

const wasmfile = process.argv[2];
if (!fs.existsSync(wasmfile)) {
    console.log("Error: File not found:", wasmfile);
    process.exit(1);
}

const wasmBuffer = fs.readFileSync(wasmfile);

async function main() {

    let memory = new ArrayBuffer(0) // will be changed after instantiate

    const captured_output = [];

    const imports = {
        env: {
            __log_utf8: (ptr, size) => {
                const str = dec.decode(new DataView(memory, ptr, size));
                captured_output.push(str);
                console.log(str);
            }
        }
    };

    const wasmModule = await WebAssembly.instantiate(wasmBuffer, imports);
    memory = wasmModule.instance.exports.memory.buffer;

    const start = wasmModule.instance.exports.start;
    const return_code = start();

    console.log("Return-Code:", return_code);

    if (return_code !== 0) {
        console.error("Expected return code 0");
        process.exit(return_code);
    }

    const expected_output = [
        '`r#try` called with ptr 0x1234',
        'Dropped',
        'Caught something!',
        '  data     : 0x1234',
        '  exception: "index out of bounds: the len is 1 but the index is 4"',
        'This program terminates correctly.',
    ];
    
    assert_equal(captured_output, expected_output);
}

function assert_equal(captured_output, expected_output) {
    if (captured_output.length != expected_output.length) {
        console.error("Unexpected number of output lines. Got", captured_output.length, "but expected", expected_output.length);
        process.exit(1); // exit with error
    }

    for (let idx = 0; idx < expected_output.length; ++idx) {
        if (captured_output[idx] !== expected_output[idx]) {
            console.error("Unexpected output");
            console.error("[got]     ", captured_output[idx]);
            console.error("[expected]", expected_output[idx]);
            process.exit(2); // exit with error
        }
    }
}

await main();