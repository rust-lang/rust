rem Because incremental builds are broken and Cargo clean does not work, we delete the build directory and rebuild
rmdir .\build\x86_64-pc-windows-msvc /s /q
set TARGET_CC=C:\ST\STM32CubeIDE_1.9.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.10.3-2021.10.win32_1.0.0.202111181127\tools\bin\arm-none-eabi-gcc.exe
python x.py build --stage 1 library --target=thumbv7em-none-eabihf
rustup toolchain link stage1 build\x86_64-pc-windows-msvc\stage1
python x.py build --stage 2 library --target=thumbv7em-none-eabihf
