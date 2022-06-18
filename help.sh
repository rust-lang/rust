for filename in ./build/aarch64-apple-darwin/llvm/build/lib/*.a; do
  lipo "$filename" -thin arm64 -output "${filename/.a/-arm64.a}" && mv "$filename" "${filename/.a/.a-UNIVERSAL}" && mv "${filename/.a/-arm64.a}" "$filename"
done
