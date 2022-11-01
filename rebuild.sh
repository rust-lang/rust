# Because incremental builds are broken and Cargo clean does not work, we delete the build directory and rebuild
rm -r ./build/x86_64-unknown-linux-gnu
python3 x.py build --stage 1 library --target=thumbv7em-none-eabihf
rustup toolchain link stage1 build/x86_64-unknown-linux-gnu/stage1
python3 x.py build --stage 2 library --target=thumbv7em-none-eabihf
