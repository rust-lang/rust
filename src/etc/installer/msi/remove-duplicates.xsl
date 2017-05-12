<?xml version="1.0" ?>
<xsl:stylesheet version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:wix="http://schemas.microsoft.com/wix/2006/wi">
    <!-- Copy all attributes and elements to the output. -->
    <xsl:template match="@*|*">
        <xsl:copy>
            <xsl:apply-templates select="@*|*"/>
        </xsl:copy>
    </xsl:template>
    <xsl:output method="xml" indent="yes" />

    <!-- LICENSE* files are installed from rustc dir. -->
    <xsl:key name="duplicates-cmp-ids" match="wix:Component[./wix:File[contains(@Source, 'LICENSE')]|./wix:File[contains(@Source, 'rust-installer-version')]]" use="@Id" />
    <xsl:template match="wix:Component[key('duplicates-cmp-ids', @Id)]" />
    <xsl:template match="wix:ComponentRef[key('duplicates-cmp-ids', @Id)]" />

    <xsl:template match="wix:File[contains(@Source, 'README.md')]">
        <xsl:copy>
            <xsl:apply-templates select="@*|*"/>
            <xsl:attribute name="Name">README-CARGO.md</xsl:attribute>
        </xsl:copy>
    </xsl:template>
</xsl:stylesheet>
