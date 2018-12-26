set -ex

export ANDROID_HOME=/android/sdk
PATH=$PATH:"${ANDROID_HOME}/tools/bin"

download_sdk() {
    mkdir -p /android
    curl -fo sdk.zip "https://dl.google.com/android/repository/sdk-tools-linux-$1.zip"
    unzip -q sdk.zip -d "$ANDROID_HOME"
    rm -f sdk.zip
}

download_sysimage() {
    abi=$1
    api=$2

    # See https://developer.android.com/studio/command-line/sdkmanager.html for
    # usage of `sdkmanager`.
    #
    # The output from sdkmanager is so noisy that it will occupy all of the 4 MB
    # log extremely quickly. Thus we must silence all output.
    yes | sdkmanager --licenses > /dev/null
    sdkmanager platform-tools emulator \
        "platforms;android-$api" \
        "system-images;android-$api;default;$abi" > /dev/null
}

create_avd() {
    abi=$1
    api=$2

    # See https://developer.android.com/studio/command-line/avdmanager.html for
    # usage of `avdmanager`.
    echo no | avdmanager create avd \
        -n "$abi-$api" \
        -k "system-images;android-$api;default;$abi"
}

download_and_create_avd() {
    download_sdk $1
    download_sysimage $2 $3
    create_avd $2 $3
}

# Usage:
#
#       setup_android_sdk 4333796 armeabi-v7a 18
#
# 4333796 =>
#   SDK tool version.
#   Copy from https://developer.android.com/studio/index.html#command-tools
# armeabi-v7a =>
#   System image ABI
# 18 =>
#   Android API Level (18 = Android 4.3 = Jelly Bean MR2)
