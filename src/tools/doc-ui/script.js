// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const puppeteer = require('puppeteer');
const fs = require('fs');
const execFileSync = require('child_process').execFileSync;
const PNG = require('png-js');

const TEST_FOLDER = 'src/test/doc-ui/';


function loadContent(content) {
    var Module = module.constructor;
    var m = new Module();
    m._compile(content, "tmp.js");
    return m.exports;
}

function readFile(filePath) {
    return fs.readFileSync(filePath, 'utf8');
}

function comparePixels(img1, img2) {
    return img1.equals(img2);
}

function main(argv) {
    /*if (argv.length !== 4) {
        console.error("Expected 2 arguments, received " + (argv.length - 2) + ":");
        console.error(" - first argument : toolchain to check as argument (for example " +
                      "'x86_64-apple-darwin')");
        console.error(" - second argument: browser path (chrome or chromium)");
        process.exitCode = 1; // exiting
    }*/
    var toolchain = argv[2];
    var stage = argv[3];

    if (stage != "0" && stage != "1" && stage != "2") {
        console.error("second argument (stage) should be either '0', '1' or '2'");
        process.exitCode = 1; // exiting
    }

    var browserPath = undefined;

    if (argv.length > 4) {
        browserPath = argv[4];
        try {
            fs.accessSync(browserPath, fs.constants.X_OK);
        } catch (err) {
            console.error('"' + browserPath + '" is not executable! Aborting.');
            process.exitCode = 2; // exiting
        }
    }

    var outPath = "build/" + toolchain + "/stage" + stage;
    execFileSync(outPath + "/bin/rustdoc",
                 ["src/test/doc-ui/lib/lib.rs", "-o", outPath + "/doc-ui"], {},
                 function(error, stdout, stderr) {
        if (error) {
            console.error(error);
            process.exitCode = 1; // exiting
        }
    });

    var imageFolder = outPath + "/doc-ui/images/";
    fs.mkdir(imageFolder, function() {});

    console.log("=> Starting doc-ui tests...");

    var loaded = [];
    var failures = 0;
    var output = [];
    fs.readdirSync(TEST_FOLDER).forEach(function(file) {
        var fullPath = TEST_FOLDER + file;
        if (file.endsWith(".js") && fs.lstatSync(fullPath).isFile()) {
            var content = readFile(fullPath) + 'exports.TEST = TEST;';
            loaded.push([file, loadContent(content).TEST]);
        }
    });

    puppeteer.launch({executablePath: browserPath}).then(async browser => {
        var docPath = 'file://' + process.cwd() + '/' + outPath + '/doc-ui/lib/';
        for (var i = 0; i < loaded.length; ++i) {
            process.stdout.write(loaded[i][0] + "... ");
            try {
                if (!('path' in loaded[i][1])) {
                    failures += 1;
                    console.log('FAILED (missing "path" key from test file)');
                    continue;
                }
                if (typeof loaded[i][1]['path'] !== "string" || loaded[i][1]['path'].length < 1) {
                    failures += 1;
                    console.log('FAILED (invalid "path" key from test file)');
                    continue;
                }
                const page = await browser.newPage();
                await page.goto(docPath + loaded[i][1]['path']);
                await page.waitFor(5000);
                var newImage = imageFolder + loaded[i][0].replace(".js", ".png");
                await page.screenshot({
                    path: newImage,
                    fullPage: true,
                });
                var originalImage = TEST_FOLDER + "images/" + loaded[i][0].replace(".js", ".png");
                if (fs.existsSync(originalImage) === false) {
                    console.log('ignored ("' + originalImage + '" not found)');
                    continue;
                }
                if (comparePixels(PNG.load(newImage).imgData,
                                  PNG.load(originalImage).imgData) === false) {
                    failures += 1;
                    console.log('FAILED (images "' + newImage + '" and "' + originalImage +
                                '" are different)');
                    continue;
                }
            } catch (err) {
                failures += 1;
                console.log("FAILED");
                output.push(loaded[i][0] + " output:\n" + err + '\n');
                continue;
            }
            console.log("ok");
        }
        if (failures > 0) {
            console.log("\n=== ERROR OUTPUT ===\n");
            console.log(output.join("\n"));
        }
        await browser.close();
        console.log("<= doc-ui tests done: " + (loaded.length - failures) + " succeeded, " +
                    failures + " failed");
        process.exitCode = failures; // exiting
    });
}

main(process.argv);
